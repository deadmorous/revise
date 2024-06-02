/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

#include "ws_sendframe/FrameServer.hpp"
#include "s3vs/VsControllerFrameOutputRW.hpp"
#include "clamp.hpp"
#include <QWebSocketServer>
#include <QWebSocket>
#include <QImage>
#include <QBuffer>
#include <QTimer>

#include <unordered_map>
#include <mutex>
#include "ipc/SharedMemory.h"

// deBUG, TODO: Comment out
// #define FRAME_SERVER_DEBUG_LOGGING

#ifdef FRAME_SERVER_DEBUG_LOGGING
#include <fstream>
#endif // FRAME_SERVER_DEBUG_LOGGING

#include "binary_io.hpp"

namespace
{

using time_point = std::chrono::time_point<std::chrono::steady_clock>;

inline time_point now() {
    return std::chrono::steady_clock::now();
}

inline double msecPassed(const time_point& from, const time_point& to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

#ifdef FRAME_SERVER_DEBUG_LOGGING
struct DebugLogData
{
    std::ofstream logFile;
    time_point startTime;

    static DebugLogData& get()
    {
        static DebugLogData data {
            std::ofstream("logs/FrameServer_debug.log"),
            now()
        };
        return data;
    }
};

template<class F>
inline void debugLog(const F& f) {
    auto& d = DebugLogData::get();
    if (d.logFile.is_open()) {
        d.logFile << msecPassed(d.startTime, now()) << " ";
        f(d.logFile);
        d.logFile << std::endl;
    }
}
#else // FRAME_SERVER_DEBUG_LOGGING
template<class F>
inline void debugLog(const F&) {
}
#endif // FRAME_SERVER_DEBUG_LOGGING

using FrameNumberType = s3vs::VsControllerFrameOutputHeader::FrameNumberType;

constexpr FrameNumberType NoNumber = ~0;
constexpr FrameNumberType FrameNaNumber = ~0 - 1;

constexpr int MaxJpegQuality = 100;

QString encodeImage(const QImage &frame, int quality)
{
    QByteArray ba;
    QBuffer bu(&ba);
    frame.save(&bu, "JPG", quality);
    QString imgBase64 = ba.toBase64();
    return QString("data:image/png;base64,") + imgBase64;
}

template <class It>
void readLongIntegers(It dst, const QStringView &s, std::size_t count, QChar delimiter)
{
    int pos = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (pos < 0)
            throw std::runtime_error("pos is less than 0");
        int dpos = s.indexOf(delimiter, pos);
        auto n = dpos > 0 ? dpos - pos : -1;
        bool ok;
        *dst++ = s.mid(pos, n).toULong(&ok);
        if (!ok)
            throw std::runtime_error("Can't convert string to int");
        pos = dpos > 0 ? dpos + 1 : -1;
    }
}

template <class T>
class MovingAverage
{
public:
    MovingAverage(std::size_t size, const T& initialValue) :
      m_buf(size, initialValue),
      m_total(initialValue * size)
    {}

    void add(const T& value)
    {
        m_total += (value - m_buf[m_index]);
        m_buf[m_index] = value;
        m_index = (m_index + 1) % m_buf.size();
    }

    T average() const {
        return m_total / m_buf.size();
    }

private:
    std::vector<T> m_buf;
    T m_total;
    std::size_t m_index = 0;
};

class AdaptiveImageQualityController
{
public:
    explicit AdaptiveImageQualityController() :
      m_overflowAverage(OverflowAverageBufSize, 0)
    {}

    void addOverflow(bool overflow)
    {
        auto overflowValue = overflow? 1.: 0.;
        m_overflowAverage.add(overflowValue);
        auto overflowAvgValue = m_overflowAverage.average();

        if (overflowAvgValue > OverflowThreshold)
            m_currentQuality -= qualityStep;
        else if (overflowAvgValue < OverflowThreshold - OverflowDelta)
            m_currentQuality += qualityStep;
        clamp(m_currentQuality, minQuality, maxQuality);
    }

    int recommendedQuality() const {
        return static_cast<int>(m_currentQuality);
    }

private:
    double m_currentQuality = MaxJpegQuality;

    MovingAverage<double> m_overflowAverage;

    static constexpr std::size_t OverflowAverageBufSize = 20;
    static constexpr double OverflowThreshold = 0.5;
    static constexpr double OverflowDelta = 0.1;
    static constexpr double qualityStep = 1;
    static constexpr double minQuality = 10;
    static constexpr double maxQuality = MaxJpegQuality;
};

} // anonymous namespace



class FrameServer::Impl
{
public:
    explicit Impl(unsigned short port) :
        m_wsServer(QStringLiteral("ws_sendframe"), QWebSocketServer::NonSecureMode),
        m_sourceNaImage(":/ws_sendframe/source_na.png"),
        m_frameSender(std::make_unique<TextFrameSender>())
    {
        QObject::connect(&m_wsServer, &QWebSocketServer::newConnection, [this]() {
            auto socket = m_wsServer.nextPendingConnection();
            Q_ASSERT(socket);
            auto clientInfo = new ClientInfo(socket);
            m_clients.push_front(clientInfo);
            auto clientListIter = m_clients.begin();
            QObject::connect(socket, &QWebSocket::textMessageReceived, [this, clientInfo](auto &message) {
                processTextMessage(*clientInfo, message);
            });
            QObject::connect(socket, &QWebSocket::disconnected, [clientInfo, clientListIter, this]() {
                clientInfo->socket->disconnect();
                delete clientInfo->socket;
                delete clientInfo;
                m_clients.erase(clientListIter);
            });
        });
        if (!m_wsServer.listen(QHostAddress::Any, port))
            throw std::runtime_error("FrameServer::start(): Failed to start listening");
        QObject::connect(&m_timer, &QTimer::timeout, [this]() { onTimerEvent(); });
        constexpr int TimerIntervalMsec = 10;
        m_timer.start(TimerIntervalMsec);
    }

    void onTimerEvent() {
        constexpr double SendFrameIntervalMsec = 40;
        constexpr int MaxPendingFrames = 3;
        auto timeNow = now();
        for (auto clientInfo : m_clients) {
            if (clientInfo->sourceId.empty())
                continue;   // No source
            if (msecPassed(clientInfo->lastSentFrameTime, timeNow) < SendFrameIntervalMsec)
                continue;   // Send next frame later
            if (clientInfo->lastConfirmedServerFrameNumber + MaxPendingFrames <= clientInfo->lastFrameNum) {
                clientInfo->qualityController.addOverflow(1);
                debugLog([&](auto& s) {
                    s << "onTimerEvent src=" << clientInfo->sourceId << " OVERFLOW";
                });
                continue;   // Too many frames sent without being confirmed
            }

            // Send next frame
            debugLog([&](auto& s) {
                s << "onTimerEvent src=" << clientInfo->sourceId << " (maybe) sending frame";
            });
            clientInfo->qualityController.addOverflow(0);
            sendFrame(*clientInfo, false);
            clientInfo->lastSentFrameTime = timeNow;
        }
    }

    ~Impl()
    {
        if (m_wsServer.isListening())
            m_wsServer.close();
    }

    void setSourceInfo(const std::string &sourceId, int shmid, const s3dmm::Vec2u &frameSize)
    {
        std::lock_guard g(m_mutex);
        auto &source = m_sources[sourceId];
        source.shmem = SharedMemory::attach(shmid);
        source.id = sourceId;
        source.frameSize = frameSize;
    }

    void removeSource(const std::string &sourceId)
    {
        std::lock_guard g(m_mutex);
        auto it = m_sources.find(sourceId);
        if (it == m_sources.end())
            throw std::range_error("FrameServer::removeSource() failed: invalid source id");
        m_sources.erase(it);
    }

private:
    std::mutex m_mutex;
    QWebSocketServer m_wsServer;
    QTimer m_timer;
    struct SourceInfo
    {
        std::string id;
        s3dmm::Vec2u frameSize;
        SharedMemory shmem;
    };
    std::unordered_map<std::string, SourceInfo> m_sources;
    QImage m_sourceNaImage;

    struct ClientInfo
    {
        ClientInfo(QWebSocket *socket) : socket(socket) {}

        QWebSocket *const socket = nullptr;
        std::string sourceId;
        mutable FrameNumberType lastFrameNum = 0;
        mutable AdaptiveImageQualityController qualityController;

        // This is for use as a client cache
        mutable std::vector<uchar> imageBuffer;
        mutable s3dmm::Vec2u frameSize;
        time_point lastSentFrameTime;
        mutable FrameNumberType lastSentShmemFrameNumber = NoNumber;
        mutable FrameNumberType lastConfirmedServerFrameNumber = NoNumber;
    };
    std::list<ClientInfo*> m_clients;

    void processTextMessage(ClientInfo &clientInfo, const QString &message)
    {
        if (message.size() > 2 && message[1] == ':')
        {
            switch (message[0].toLatin1())
            {
            case 's':
            {
                clientInfo.sourceId = message.mid(2).toStdString();
                std::unique_lock lk(m_mutex);
                auto ok = m_sources.find(clientInfo.sourceId) != m_sources.end();
                lk.unlock();
                clientInfo.socket->sendTextMessage(ok ? "Ok" : "Unknown source");
                break;
            }
            case 'n':
            {
                // [0] - frame number in shmem, [1] - frame number on server
                FrameNumberType frame_numbers[2] = { NoNumber, NoNumber };
                debugLog([&](auto& s) {
                    s << "processTextMessage: src=" << clientInfo.sourceId << " reply received: " << message.toStdString();
                });
                try {
                    readLongIntegers(frame_numbers, QStringView(message).sliced(2), 2, ':');
                    clientInfo.lastConfirmedServerFrameNumber = frame_numbers[1];
                }
                catch(std::exception&) {
                }
                break;
            }
            }
        }
        else if (message.size() == 1) {
            switch (message[0].toLatin1()) {
            case 'F':
                sendFrame(clientInfo, true);
                break;
            case 'B':
                m_frameSender = std::make_unique<BinaryFrameSender>();
                clientInfo.socket->sendTextMessage("Ok");
                break;
            case 'T':
                m_frameSender = std::make_unique<TextFrameSender>();
                clientInfo.socket->sendTextMessage("Ok");
                break;
            }
        }
    }



    struct FrameSenderInterface
    {
        virtual ~FrameSenderInterface() = default;

        virtual void sendRenderedImageFrame(
            const ClientInfo &clientInfo,
            const s3vs::VsControllerFrameOutputHeader& hdr,
            int quality,
            char type = 'n') = 0;

        void sendFullQualityRenderedImageFrame(
            const ClientInfo &clientInfo,
            const s3vs::VsControllerFrameOutputHeader& hdr)
        {
            sendRenderedImageFrame(clientInfo, hdr, MaxJpegQuality, 'F');
        }

    };
    std::unique_ptr<FrameSenderInterface> m_frameSender;

    class TextFrameSender : public FrameSenderInterface
    {
    public:
        static QString frameHeaderToString(
            const s3vs::VsControllerFrameOutputHeader& hdr, FrameNumberType frameServerFrameNum)
        {
            auto result = "##" + QString::number(hdr.frameNumber) +
                          "##" + QString::number(frameServerFrameNum) +
                          "##" + QString::number(hdr.renderingDuration.level) +
                          "##" + QString::number(hdr.renderingDuration.durationMs);
            return result;
        }

        void sendRenderedImageFrame(
            const ClientInfo &clientInfo,
            const s3vs::VsControllerFrameOutputHeader& hdr,
            int quality,
            char type = 'n') override
        {
            debugLog([&](auto& s) {
                s << "sendRenderedImageFrame src=" << clientInfo.sourceId;
            });
            auto w = clientInfo.frameSize[0];
            auto h = clientInfo.frameSize[1];
            sendEncodedImageFrame(
                clientInfo, hdr,
                encodeImage(
                    QImage(clientInfo.imageBuffer.data(), w, h, w<<2, QImage::Format_RGB32),
                    quality), quality, type);
            clientInfo.lastSentShmemFrameNumber = hdr.frameNumber;
        }

    private:
        static void sendEncodedImageFrame(
            const ClientInfo &clientInfo,
            const s3vs::VsControllerFrameOutputHeader& hdr,
            const QString& encodedImage, int quality, char type)
        {
            auto hdrText = type + frameHeaderToString(hdr, clientInfo.lastFrameNum) +
                           "##" + QString::number(quality);
            debugLog([&](auto& s) {
                s << "sendEncodedImageFrame: src=" << clientInfo.sourceId << " " << hdrText.toStdString()
                  << " " << clientInfo.frameSize << " size=" << encodedImage.size();
            });
            clientInfo.socket->sendTextMessage(hdrText + "##" + encodedImage);
            ++clientInfo.lastFrameNum;
        }
    };

    class BinaryFrameSender : public FrameSenderInterface
    {
    public:
        void sendRenderedImageFrame(
            const ClientInfo &clientInfo,
            const s3vs::VsControllerFrameOutputHeader& hdr,
            int quality,
            char type = 'n') override
        {
            debugLog([&](auto& s) {
                s << "sendRenderedImageFrame src=" << clientInfo.sourceId;
            });
            QBuffer buf;
            buf.open(QIODevice::WriteOnly);
            using BW = s3dmm::BinaryWriterTemplate<QIODevice, std::uint32_t, std::uint32_t>;
            BW writer(buf);
            writer
                    << type
                    << hdr.frameNumber
                    << clientInfo.lastFrameNum
                    << hdr.renderingDuration.level
                    << hdr.renderingDuration.durationMs
                    << quality;
            auto w = clientInfo.frameSize[0];
            auto h = clientInfo.frameSize[1];
            QImage img(clientInfo.imageBuffer.data(), w, h, w<<2, QImage::Format_RGB32);
            img.save(&buf, "JPG", quality);
            clientInfo.socket->sendBinaryMessage(buf.buffer());
            clientInfo.lastSentShmemFrameNumber = hdr.frameNumber;
            ++clientInfo.lastFrameNum;
        }
    };

    void sendFrame(
        const ClientInfo &clientInfo, bool fullQuality)
    {
        std::unique_lock g(m_mutex);
        auto it = m_sources.find(clientInfo.sourceId);
        if (it == m_sources.end())
        {
            g.unlock();
            s3vs::VsControllerFrameOutputHeader hdr{
                FrameNaNumber,
                {0, 0}
            };
            if (clientInfo.lastSentShmemFrameNumber != FrameNaNumber) {
                unsigned int w = m_sourceNaImage.width();
                unsigned int h = m_sourceNaImage.height();
                auto frameSizeInBytes = (w*h) << 2;
                clientInfo.frameSize = { w, h };
                clientInfo.imageBuffer.resize(frameSizeInBytes);
                m_frameSender->sendRenderedImageFrame(
                    clientInfo, hdr, MaxJpegQuality, fullQuality? 'F': 'n');
            }
        }
        else
        {
            auto &sourceInfo = it->second;
            auto w = sourceInfo.frameSize[0];
            auto h = sourceInfo.frameSize[1];
            auto frameSizeInBytes = (w*h) << 2;
            clientInfo.frameSize = { w, h };
            clientInfo.imageBuffer.resize(frameSizeInBytes);
            s3vs::VsControllerFrameOutputRW rw(&sourceInfo.shmem, frameSizeInBytes);
            s3vs::VsControllerFrameOutputHeader hdr;
            rw.readFrame(hdr, clientInfo.imageBuffer.data(), [&](const s3vs::VsControllerFrameOutputHeader& hdr) {
                return hdr.frameNumber != clientInfo.lastSentShmemFrameNumber;
            });
            g.unlock();
            if (fullQuality)
                m_frameSender->sendFullQualityRenderedImageFrame(
                    clientInfo, hdr);
            else if (hdr.frameNumber != clientInfo.lastSentShmemFrameNumber)
                m_frameSender->sendRenderedImageFrame(
                    clientInfo, hdr, clientInfo.qualityController.recommendedQuality());
        }
    }
};

FrameServer::FrameServer(unsigned short port) : m_impl(std::make_unique<Impl>(port))
{
}

FrameServer::~FrameServer() = default;

void FrameServer::setSourceInfo(
        const std::string &sourceId, int shmid, const s3dmm::Vec2u &frameSize)
{
    m_impl->setSourceInfo(sourceId, shmid, frameSize);
}

void FrameServer::removeSource(const std::string &sourceId)
{
    m_impl->removeSource(sourceId);
}
