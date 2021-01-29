#include "s3vs/VsWorkerTimestamps.hpp"
#include "foreach_byindex32.hpp"

#include <QGuiApplication>
#include <QFontDatabase>
#include <QImage>
#include <QPainter>
#include <QDir>

#include <fstream>
#include <iostream>
#include <regex>

#include <boost/program_options.hpp>
#include <boost/range/algorithm/remove_if.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/lexical_cast.hpp>

#include "silver_bullets/enum_names.hpp"
#include "silver_bullets/fs_ns_workaround.hpp"
#include "enumHelp.hpp"

using namespace std;
using namespace s3dmm;
using namespace s3vs;

enum class Mode
{
    Summary,
    MinMaxTaskDuration,
    MinMaxTaskStageDuration,
    TaskTimeline,
    TaskStageTimeline
};

SILVER_BULLETS_BEGIN_DEFINE_ENUM_NAMES(Mode)
    { Mode::Summary, "summary" },
    { Mode::MinMaxTaskDuration, "task_duration" },
    { Mode::MinMaxTaskStageDuration, "task_stage_duration" },
    { Mode::TaskTimeline, "task_timeline" },
    { Mode::TaskStageTimeline, "task_stage_timeline" }
SILVER_BULLETS_END_DEFINE_ENUM_NAMES()

struct Param
{
    Mode mode;
    string inputFileName;
    string frameAveraging = "none";
    string projString;
    int rowHeight = 10;
};

struct AveragingParam
{
public:
    AveragingParam(size_t totalCount, const string& param)
    {
        if (param != "none") {
            regex rx("(\\d+)-(\\d+|\\$)");
            smatch m;
            if (!regex_match(param, m, rx))
                throw runtime_error("AveragingParam: invalid parameter '" + param + "'");
            m_firstNumber = boost::lexical_cast<size_t>(m[1]);
            if (m[2] == "$")
                m_lastNumber = totalCount - 1;
            else {
                m_lastNumber = boost::lexical_cast<size_t>(m[2]);
                if (m_lastNumber < m_firstNumber)
                    throw runtime_error("AveragingParam: invalid parameter '" + param + "' - last < first");
                if (m_lastNumber >= totalCount)
                    throw runtime_error("AveragingParam: invalid parameter '" + param + "' - out of range");
            }
            m_enabled = totalCount > 0;
        }
    }

    bool enabled() const {
        return m_enabled;
    }

    size_t firstNumber() const {
        return m_firstNumber;
    }

    size_t lastNumber() const {
        return m_lastNumber;
    }

private:
    bool m_enabled = false;
    size_t m_firstNumber = 0;
    size_t m_lastNumber = 0;
};

struct TS : VsWorkerTimestampsTemplate<double>
{
    unsigned int node = ~0;
    unsigned int threadId = ~0;

    double duration() const {
        return afterClip - start;
    }

    static unsigned int toLevel(const TS& ts) {
        return ts.level;
    }

    static unsigned int toThreadId(const TS& ts) {
        return ts.threadId;
    }

    static unsigned int toNodeId(const TS& ts) {
        return ts.node;
    }

    static double toDuration(const TS& ts) {
        return ts.duration();
    }
};

template<class F, class T>
inline auto make_equal_to(F f, const T& value) {
    return [f, value](const auto& x) {
        return f(x) == value;
    };
}

template<class F>
inline auto make_less(F f) {
    return [f](const auto& a, const auto& b) {
        return f(a) < f(b);
    };
}

inline std::istream& operator>>(std::istream& s, TS& t)
{
    s >> t.node >> t.threadId;
    VsWorkerTimestampsTemplate<double>& base = t;
    s >> base;
    return s;
}

struct TimeBar
{
    enum Type {
        ClearViewport,
        InitBlockSorter,
        BlockReadSparsePrimaryField,
        BlockComputeDensePrimaryField,
        BlockGetPrimaryFieldRange,
        BlockReadSparseSecondaryField,
        BlockComputeDenseSecondaryField,
        BlockGetSecondaryFieldRange,
        BlockRender,
        Download,
        GetClipRect,
        Clip,
        End
    };

    double time;
    Type type;
};

SILVER_BULLETS_BEGIN_DEFINE_ENUM_NAMES(TimeBar::Type)
    { TimeBar::ClearViewport, "ClearViewport" },
    { TimeBar::InitBlockSorter, "InitBlockSorter" },
    { TimeBar::BlockReadSparsePrimaryField, "BlockReadSparsePrimaryField" },
    { TimeBar::BlockComputeDensePrimaryField, "BlockComputeDensePrimaryField" },
    { TimeBar::BlockGetPrimaryFieldRange, "BlockGetPrimaryFieldRange" },
    { TimeBar::BlockReadSparseSecondaryField, "BlockReadSparseSecondaryField" },
    { TimeBar::BlockComputeDenseSecondaryField, "BlockComputeDenseSecondaryField" },
    { TimeBar::BlockGetSecondaryFieldRange, "BlockGetSecondaryFieldRange" },
    { TimeBar::BlockRender, "BlockRender" },
    { TimeBar::Download, "Download" },
    { TimeBar::GetClipRect, "GetClipRect" },
    { TimeBar::Clip, "Clip" },
    { TimeBar::End, "End" }
SILVER_BULLETS_END_DEFINE_ENUM_NAMES()

std::vector<TimeBar> toTimeBars(const TS& ts)
{
    std::vector<TimeBar> result;
    result.push_back({ts.start, TimeBar::ClearViewport});
    result.push_back({ts.afterClearViewport, TimeBar::InitBlockSorter});
    result.push_back({ts.afterSortBlocks, TimeBar::BlockReadSparsePrimaryField});

    foreach_byindex32(iblock, ts.blocks) {
        auto& block = ts.blocks[iblock];
        result.push_back({block.afterPrimaryField.afterReadSparseField, TimeBar::BlockComputeDensePrimaryField});
        result.push_back({block.afterPrimaryField.afterComputeDenseField, TimeBar::BlockGetPrimaryFieldRange});
        result.push_back({block.afterPrimaryField.afterGetFieldRange, TimeBar::BlockReadSparseSecondaryField});
        result.push_back({block.afterSecondaryField.afterReadSparseField, TimeBar::BlockComputeDenseSecondaryField});
        result.push_back({block.afterSecondaryField.afterComputeDenseField, TimeBar::BlockGetSecondaryFieldRange});
        result.push_back({block.beforeRender, TimeBar::BlockRender});
        if (iblock+1 < containerSize)
            result.push_back({block.afterRender, TimeBar::BlockReadSparsePrimaryField});
        else
            result.push_back({block.afterRender, TimeBar::Download});
    }
    result.push_back({ts.afterRenderDenseField, TimeBar::GetClipRect});
    result.push_back({ts.afterGetClipRect, TimeBar::Clip});
    result.push_back({ts.afterClip, TimeBar::End});
    return result;
}

std::vector<TimeBar> timeBarDurations(const std::vector<TimeBar>& timeBars)
{
    BOOST_ASSERT(!timeBars.empty() && timeBars.back().type == TimeBar::End);
    std::vector<TimeBar> result(timeBars.size() - 1);
    foreach_byindex32_from(i, 1u, timeBars) {
        auto& dst = result[i-1];
        auto& src = timeBars[i-1];
        dst.time = timeBars[i].time - src.time;
        dst.type = src.type;
    }
    return result;
}

QColor timeBarColor(const TimeBar::Type& type)
{
    switch(type) {
        case TimeBar::ClearViewport:
            return QColor::fromRgba(0xff2b8a00);
        case TimeBar::InitBlockSorter:
            return QColor::fromRgba(0xffff8a00);
        case TimeBar::BlockReadSparsePrimaryField:
            return QColor::fromRgba(0x88ff0000);
        case TimeBar::BlockComputeDensePrimaryField:
            return QColor::fromRgba(0x880000ff);
        case TimeBar::BlockGetPrimaryFieldRange:
            return QColor::fromRgba(0xffa4ff12);
        case TimeBar::BlockReadSparseSecondaryField:
            return QColor::fromRgba(0x88aa0000);
        case TimeBar::BlockComputeDenseSecondaryField:
            return QColor::fromRgba(0x880000aa);
        case TimeBar::BlockGetSecondaryFieldRange:
            return QColor::fromRgba(0xff655507);
        case TimeBar::BlockRender:
            return QColor::fromRgba(0x8800ffff);
        case TimeBar::Download:
            return QColor::fromRgba(0x8800ff00);
        case TimeBar::GetClipRect:
            return QColor::fromRgba(0xffffcc00);
        case TimeBar::Clip:
            return QColor::fromRgba(0xffff00ff);
        case TimeBar::End:
            return QColor::fromRgba(0xffffffff);
    }
    Q_ASSERT(false);
    return Qt::black;
}



bool isTimebarSequenceMonotonous(const std::vector<TimeBar>& tbars)
{
    return std::is_sorted(tbars.begin(), tbars.end(), [](const TimeBar& a, const TimeBar& b) {
        return a.time < b.time;
    });
}



class Stats
{
public:
    void add(double x)
    {
        if (m_count == 0)
            m_min = m_max = x;
        else {
            if (m_min > x)
                m_min = x;
            else if (m_max < x)
                m_max = x;
        }
        m_sum += x;
        ++m_count;
    }

    Stats& operator<<(double x)
    {
        add(x);
        return *this;
    }

    size_t count() const {
        return m_count;
    }

    double min() const {
        return m_min;
    }

    double max() const {
        return m_max;
    }

    double average() const {
        return m_sum / m_count;
    }

private:
    size_t m_count = 0;
    double m_min = 0;
    double m_max = 0;
    double m_sum = 0;
};



struct Table
{
    vector<string> headers;
    vector<vector<double>> rows;
};

inline ostream& operator<<(ostream& s, const Table& table)
{
    TsvWriter w(s);
    for (auto& h : table.headers)
        w << h;
    w << endl;
    for (auto& row : table.rows) {
        for (auto& x : row)
            w << x;
        w << endl;
    }
    return s;
}

inline ostream& operator<<(ostream& s, const vector<Table>& tables)
{
    for (auto& table : tables)
        s << table << endl;
    return s;
}



Table averageTables(const vector<Table>& tables, const AveragingParam& avp)
{
    Table result;
    result.headers = tables.front().headers;
    for (auto& table : tables) {
        result.rows.emplace_back();
        auto& row = result.rows.back();
        foreach_byindex32(icol, table.rows.front()) {
            Stats st;
            for (size_t irow=avp.firstNumber(); irow<=avp.lastNumber(); ++irow)
                st << table.rows.at(irow).at(icol);
            row.push_back(st.average());
        }
    }
    return result;
}

template<class T>
vector<T> subArray(const vector<T>& v, const vector<size_t>& indices)
{
    vector<T> result(indices.size());
    foreach_byindex32(i, indices)
        result[i] = v[indices[i]];
    return result;
}

Table projectTable(const Table& table, const string& pattern)
{
    if (pattern.empty())
        return table;
    vector<size_t> matchingColumns;
    regex rx(pattern);
    foreach_byindex32(icol, table.headers) {
        if (regex_match(table.headers[icol], rx))
            matchingColumns.push_back(icol);
    }
    Table result;
    result.headers = subArray(table.headers, matchingColumns);
    for (auto& row : table.rows)
        result.rows.push_back(subArray(row, matchingColumns));
    return result;
}

vector<Table> projectTables(const vector<Table>& tables, const string& pattern)
{
    if (pattern.empty())
        return tables;
    vector<Table> result(tables.size());
    foreach_byindex32(itable, tables)
        result[itable] = projectTable(tables[itable], pattern);
    return result;
}

vector<Table> limitTables(const Param& param, const vector<Table>& tables)
{
    vector<Table> result = tables;
    AveragingParam avp(tables.front().rows.size(), param.frameAveraging);
    if (avp.enabled())
        result = { averageTables(result, avp) };
    if (!param.projString.empty())
        result = projectTables(result, param.projString);
    return result;
}



void checkTimestamp(const TS& ts)
{
    auto tlast = ts.start;
    auto check = [&tlast](double t) {
        if (t < tlast)
            throw std::runtime_error("Invalid timestamp (non-monotonous)");
        tlast = t;
    };
    check(ts.afterClearViewport);
    check(ts.afterSortBlocks);
    foreach_byindex32(iblock, ts.blocks) {
        check(ts.blocks[iblock].afterPrimaryField.afterReadSparseField);
        check(ts.blocks[iblock].afterPrimaryField.afterComputeDenseField);
        check(ts.blocks[iblock].afterPrimaryField.afterGetFieldRange);
        check(ts.blocks[iblock].afterSecondaryField.afterReadSparseField);
        check(ts.blocks[iblock].afterSecondaryField.afterComputeDenseField);
        check(ts.blocks[iblock].afterSecondaryField.afterGetFieldRange);
        check(ts.blocks[iblock].beforeRender);
        check(ts.blocks[iblock].afterRender);
    }
    check(ts.afterRenderDenseField);
    check(ts.afterGetClipRect);
    check(ts.afterClip);
}

struct FrameTimestamps
{
public:

    const std::vector<TS> items() const {
        return m_items;
    }

    void push_back(const TS& ts) {
        m_items.push_back(ts);
    }

    double startTime() const
    {
        return boost::range::min_element(m_items, [](const TS& a, const TS& b) {
            return a.start < b.start;
        })->start;
    }

    double endTime() const
    {
        return boost::range::max_element(m_items, [](const TS& a, const TS& b) {
            return a.afterClip < b.afterClip;
        })->afterClip;
    }

    double duration() const {
        return endTime() - startTime();
    }

    unsigned int workerCount() const {
        return m_items.size();
    }

    void resolve()
    {
        auto it = boost::range::remove_if(m_items, [](const TS& ts) {
            return ts.start == 0;
        });
        m_items.resize(it - m_items.begin());
        BOOST_ASSERT(!m_items.empty());
    }

    void resolve(const FrameTimestamps& prev)
    {
        auto t0 = boost::range::max_element(prev.m_items, [](const TS& a, const TS& b) {
            return a.start < b.start;
        })->start;
        auto it = boost::range::remove_if(m_items, [t0](const TS& ts) {
            return ts.start <= t0;
        });
        m_items.resize(it - m_items.begin());
        BOOST_ASSERT(!m_items.empty());
    }

    void applyTimeShift()
    {
        auto t0 = startTime();
        auto shiftFieldTs = [t0](auto& fieldTs) {
            fieldTs.afterReadSparseField -= t0;
            fieldTs.afterComputeDenseField -= t0;
            fieldTs.afterGetFieldRange -= t0;
        };
        for (auto& item : m_items) {
            item.start -= t0;
            item.afterClearViewport -= t0;
            item.afterSortBlocks -= t0;
            for (auto& block : item.blocks) {
                shiftFieldTs(block.afterPrimaryField);
                shiftFieldTs(block.afterSecondaryField);
                block.beforeRender -= t0;
                block.afterRender -= t0;
            }
            item.afterRenderDenseField -= t0;
            item.afterGetClipRect -= t0;
            item.afterClip -= t0;
        }
    }

    size_t levelCount() const {
        return count(TS::toLevel);
    }

    size_t threadCount() const {
        return count([](auto& item) { return item.threadId; });
    }

    size_t nodeCount() const {
        return count([](auto& item) { return item.node; });
    }

    FrameTimestamps filterByLevel(unsigned int level) const {
        return filterEqual(TS::toLevel, level);
    }

    FrameTimestamps filterByThread(unsigned int threadId) const {
        return filterEqual(TS::toThreadId, threadId);
    }

    FrameTimestamps filterByNode(unsigned int nodeId) const {
        return filterEqual(TS::toNodeId, nodeId);
    }

    template<class F>
    FrameTimestamps filter(const F& f) const
    {
        FrameTimestamps result;
        std::copy_if(m_items.begin(), m_items.end(), std::back_inserter(result.m_items), f);
        return result;
    }

    template<class F, class T>
    FrameTimestamps filterEqual(const F& f, const T& value) const {
        return filter(make_equal_to(f, value));
    }

private:
    std::vector<TS> m_items;

    template<class F>
    size_t count(const F& f) const
    {
        if (m_items.empty())
            return 0;
        else
            return 1 + f(*max_element(m_items.begin(), m_items.end(), make_less(f)));
    }
};

std::vector<FrameTimestamps> readTimestamps(istream& is)
{
    std::vector<FrameTimestamps> result;
    while (true) {
        FrameTimestamps tsp;
        while (true) {
            string line;
            getline(is, line);
            if (is.fail())
                break;
            if (line.empty())
                continue;
            if (line == "-")
                break;
            TS ts;
            istringstream iss(line);
            iss >> ts;
            checkTimestamp(ts);
            tsp.push_back(ts);
        }
        if (tsp.items().empty())
            break;
        else {
            if (result.empty())
                tsp.resolve();
            else
                tsp.resolve(result.back());
            result.push_back(tsp);
        }
    }
    for (auto& tsp : result)
        tsp.applyTimeShift();
    return result;
}

class Frames
{
public:
    Frames() = default;
    explicit Frames(istream& is) : m_frames(readTimestamps(is))
    {
        for (auto& tsp : m_frames) {
            for(auto& ts : tsp.items()) {
                auto tbars = toTimeBars(ts);
                BOOST_ASSERT(isTimebarSequenceMonotonous(tbars));
            }
        }
    }

    const std::vector<FrameTimestamps>& frames() const {
        return m_frames;
    }

private:
    std::vector<FrameTimestamps> m_frames;
};

Table computeSummary(const Param&, const Frames& frameData)
{
    auto& frames = frameData.frames();
    BOOST_ASSERT(!frames.empty());

    auto& firstFrame = frames.at(0);
    auto levelCount = firstFrame.levelCount();

    return {
        { "frame_count", "level_count" },
        { { static_cast<double>(frames.size()), static_cast<double>(levelCount) } }
    };
}

vector<Table> computeMinMaxTaskDuration(const Param&, const Frames& frameData)
{
    auto& frames = frameData.frames();
    BOOST_ASSERT(!frames.empty());

    auto& firstFrame = frames.at(0);
    auto levelCount = firstFrame.levelCount();

    vector<Table> result;

    for (size_t level=0; level<levelCount; ++level) {
        result.emplace_back();
        auto& table = result.back();
        table.headers =  { "frame", "level", "duration", "task_min_duration", "task_max_duration" };
        foreach_byindex32(iframe, frames) {
            auto& frame = frames[iframe];
            auto levelTs = frame.filterByLevel(level);
            auto& items = levelTs.items();
            auto taskMinDuration = boost::range::min_element(items, make_less(TS::toDuration))->duration();
            auto taskMaxDuration = boost::range::max_element(items, make_less(TS::toDuration))->duration();
            table.rows.emplace_back(vector<double>{
                static_cast<double>(iframe),
                static_cast<double>(level),
                static_cast<double>(levelTs.duration()),
                static_cast<double>(taskMinDuration),
                static_cast<double>(taskMaxDuration)
            });
        }
    }
    return result;
}

vector<Table> computeMinMaxTaskStageDuration(const Param&, const Frames& frameData)
{
    auto& frames = frameData.frames();
    BOOST_ASSERT(!frames.empty());

    auto& firstFrame = frames.at(0);
    auto levelCount = firstFrame.levelCount();

    vector<Table> result;

    struct StatsComponent
    {
        const char *suffix;
        std::function<double(const Stats&)> func;
    };
    StatsComponent statsComponents[] = {
        { "_count", [](const Stats& stats) { return stats.count(); } },
        { "_min", [](const Stats& stats) { return stats.min(); } },
        { "_max", [](const Stats& stats) { return stats.max(); } },
        { "_avg", [](const Stats& stats) { return stats.average(); } },
    };

    for (size_t level=0; level<levelCount; ++level) {
        result.emplace_back();
        auto& table = result.back();
        table.headers = { "frame", "level", "duration" };
        for (auto& statsComponent : statsComponents) {
            for (auto it = silver_bullets::enum_item_begin<TimeBar::Type>(),
                      end = silver_bullets::enum_item_end<TimeBar::Type>()-1;
                 it != end; ++it)
            {
                table.headers.push_back(string(it->second) + statsComponent.suffix);
            }
        }

        foreach_byindex32(iframe, frames) {
            auto& frame = frames[iframe];
            auto levelTs = frame.filterByLevel(level);
            auto& items = levelTs.items();
            vector<Stats> stageStats(silver_bullets::enum_item_count<TimeBar::Type>() - 1);
            for (auto& item : items)
                for (auto& tb : timeBarDurations(toTimeBars(item)))
                    stageStats.at(tb.type) << tb.time;

            table.rows.emplace_back();
            auto& row = table.rows.back();
            row.push_back(iframe);
            row.push_back(level);
            row.push_back(levelTs.duration());
            for (auto& statsComponent : statsComponents)
                for(auto& stage : stageStats)
                    row.push_back(statsComponent.func(stage));
        }
    }

    return result;
}

void showSummary(const Param& param, const Frames& frameData) {
    cout << limitTables(param, {computeSummary(param, frameData)});
}

void showMinMaxTaskDuration(const Param& param, const Frames& frameData) {
    cout << limitTables(param, {computeMinMaxTaskDuration(param, frameData)});
}

void showMinMaxTaskStageDuration(const Param& param, const Frames& frameData) {
    cout << limitTables(param, computeMinMaxTaskStageDuration(param, frameData));
}

QString outputDirPrefix(const Param& param)
{
    using namespace filesystem;
    auto outputDir = param.inputFileName + ".dir";
    if (exists(outputDir)) {
        if (!is_directory(outputDir))
            throw runtime_error("File '" + outputDir + "' already exists and is not a directory");
    }
    else if (!create_directory(outputDir))
        throw runtime_error("Failed to create directory '" + outputDir + "'");
    auto result = QString::fromStdString(outputDir);
    if (!result.endsWith('/'))
        result += '/';
    return result;
}

void writeTaskTimeline(const Param& param, const Frames& frameData)
{
    auto outputPrefix = outputDirPrefix(param);
    constexpr int PixelsPerMs = 1; // Pixels in one millisecond
    auto xpix = [](double time) {
        return static_cast<int>(time * 1000 * PixelsPerMs);
    };
    auto ypix = [&param](int threadId) {
        return threadId * param.rowHeight;
    };

    auto& frames = frameData.frames();
    BOOST_ASSERT(!frames.empty());

    auto& firstFrame = frames.at(0);
    auto levelCount = firstFrame.levelCount();

    for (size_t level=0; level<levelCount; ++level) {
        foreach_byindex32(iframe, frames) {
            auto& frame = frames[iframe];
            auto levelTs = frame.filterByLevel(level);
            levelTs.applyTimeShift();
            auto& items = levelTs.items();
            auto imageWidth = xpix(levelTs.endTime()) + 1;
            auto imageHeight = param.rowHeight * levelTs.threadCount() + 1;
            QImage img(imageWidth, imageHeight, QImage::Format_RGB32);
            img.fill(Qt::white);
            QPainter painter(&img);
            for (auto& item : items) {
                auto x1 = xpix(item.start);
                auto x2 = xpix(item.afterClip) - 1;
                if (x2 > x1) {
                    auto w = x2 - x1;
                    auto y = ypix(item.threadId);
                    auto h = param.rowHeight;
                    if (h > 1) {
                        ++y;
                        --h;
                    }
                    painter.fillRect(x1, y, w, h, QColor::fromRgba(0x88000000));
                }
            }
            auto fileName = QString("%1task_timeline_frame-%2_level-%3.png").arg(outputPrefix).arg(iframe).arg(level);
            img.save(fileName);
        }
    }
}

void writeTaskStageTimeline(const Param& param, const Frames& frameData)
{
    auto outputPrefix = outputDirPrefix(param);
    constexpr int PixelsPerMs = 1; // Pixels in one millisecond
    auto xpix = [](double time) {
        return static_cast<int>(time * 1000 * PixelsPerMs);
    };
    auto ypix = [&param](int threadId) {
        return threadId * param.rowHeight;
    };

    auto& frames = frameData.frames();
    BOOST_ASSERT(!frames.empty());

    auto& firstFrame = frames.at(0);
    auto levelCount = firstFrame.levelCount();

    for (size_t level=0; level<levelCount; ++level) {
        foreach_byindex32(iframe, frames) {
            auto& frame = frames[iframe];
            auto levelTs = frame.filterByLevel(level);
            levelTs.applyTimeShift();
            auto& items = levelTs.items();
            auto imageWidth = xpix(levelTs.endTime()) + 1;
            auto imageHeight = param.rowHeight * levelTs.threadCount() + 1;
            QImage img(imageWidth, imageHeight, QImage::Format_RGB32);
            img.fill(Qt::white);
            QPainter painter(&img);
            for (auto& item : items) {
                auto taskTimeBars = toTimeBars(item);
                auto y = ypix(item.threadId);
                auto h = param.rowHeight;
                if (h > 1) {
                    ++y;
                    --h;
                }
                foreach_byindex32_from(i, 1u, taskTimeBars) {
                    auto& bar = taskTimeBars[i-1];
                    auto& nextBar = taskTimeBars[i];

                    auto x1 = xpix(bar.time);
                    auto x2 = xpix(nextBar.time);
                    if (x2 > x1) {
                        auto w = x2 - x1;
                        painter.fillRect(x1, y, w, h, timeBarColor(bar.type));
                    }
                }
            }
            auto fileName = QString("%1task_stage_timeline_frame-%2_level-%3.png").arg(outputPrefix).arg(iframe).arg(level);
            img.save(fileName);
        }
    }

    // Draw legend
    {
        auto RowSpacing = 20;
        auto ItemHeight = 15;
        auto ItemWidth = 20;
        auto Width = 300;
        auto ItemPos = 10;
        auto TextPos = 40;
        QImage legend(
            Width,
            RowSpacing*(silver_bullets::enum_item_count<TimeBar::Type>() - 1),
            QImage::Format_RGB32);
        legend.fill(Qt::white);
        QPainter painter(&legend);
        foreach_byindex32_from(i, 1u, silver_bullets::enum_item_range<TimeBar::Type>()) {
            auto ytop = (i-1) * RowSpacing;
            auto typeIt = silver_bullets::enum_item_begin<TimeBar::Type>() + (i-1);
            painter.fillRect(
                ItemPos, ytop + (RowSpacing - ItemHeight)/2, ItemWidth, ItemHeight,
                timeBarColor(typeIt->first));
            painter.drawText(QRect(TextPos, ytop, Width-TextPos, RowSpacing),
                             Qt::AlignLeft | Qt::AlignVCenter,
                             typeIt->second);
        }
        legend.save(outputPrefix + "task_stage_timeline_legend.png");
    }
}

void run(const Param& param)
{
    ifstream is(param.inputFileName);
    if (is.fail())
        throw runtime_error(string("Failed to open input timestamps file '") + param.inputFileName + "'");
    Frames frameData(is);
    if (frameData.frames().empty())
        throw runtime_error("No frames are found");

    switch(param.mode) {
        case Mode::Summary:
            showSummary(param, frameData);
            break;
        case Mode::MinMaxTaskDuration:
            showMinMaxTaskDuration(param, frameData);
            break;
        case Mode::MinMaxTaskStageDuration:
            showMinMaxTaskStageDuration(param, frameData);
            break;
        case Mode::TaskTimeline:
            writeTaskTimeline(param, frameData);
            break;
        case Mode::TaskStageTimeline:
            writeTaskStageTimeline(param, frameData);
            break;
    }
}

void loadFonts()
{
    static const char *fontNames[] = {
        "Pfennig.ttf",
        0
    };
    QDir dir(":/fonts");
    for (int i=0; fontNames[i]; ++i)
        if (QFontDatabase::addApplicationFont(dir.absoluteFilePath(fontNames[i])) == -1)
            throw runtime_error(string("Failed to load font '") + fontNames[i] + "'");
}

class ArgcArgv
{
public:
    ArgcArgv(int argc, char *argv[]) :
        m_args(argc),
        m_argc(-1)
    {
        transform(argv, argv + argc, m_args.begin(), [](char *s) { return s; });
    }

    void insertArgument(size_t pos, const string& arg)
    {
        m_args.insert(m_args.begin() + pos, arg);
        m_argc = -1;
    }

    int& argc() const
    {
        update();
        m_argc_buf = m_argc;
        return m_argc_buf;
    }

    char **argv() const
    {
        update();
        return m_argv.data();
    }

private:
    vector<string> m_args;
    mutable int m_argc = -1;
    mutable vector<char*> m_argv;
    mutable int m_argc_buf = -1;

    void update() const
    {
        if (m_argc == -1) {
            m_argc = m_args.size();
            m_argv.resize(m_argc+1);
            transform(m_args.begin(), m_args.end(), m_argv.begin(), [](const string& s) {
                return const_cast<char*>(s.c_str());
            });
            m_argv.back() = nullptr;
        }
    }
};

int main(int argc, char *argv[])
{
    ArgcArgv argcArgv(argc, argv);
    argcArgv.insertArgument(1, "-platform");
    argcArgv.insertArgument(2, "offscreen");
    QGuiApplication app(argcArgv.argc(), argcArgv.argv());
    try {
        loadFonts();
        Param param;
        string modeString = silver_bullets::enum_item_name(Mode::Summary);

        namespace po = boost::program_options;
        auto po_value = [](auto& x) {
            return po::value(&x)->default_value(x);
        };
        po::options_description po_generic("Gerneric options");
        po_generic.add_options()
                ("help,h", "produce help message");

        po::options_description po_main("Main parameters");
        po_main.add_options()
                ("input", po_value(param.inputFileName), "Input timestamps file name (typically named VsRendererDrawTimestamps.log)")
                ("mode", po_value(modeString), enumHelp<Mode>("Processing mode"))
                ("favg", po_value(param.frameAveraging), "Average over specified frame range (ex. 1-5, 1-$, none)")
                ("proj", po_value(param.projString), "Limit columns (regular expression to match column headers against)")
                ("ids", po_value(param.rowHeight), "The height of one row, corresp. to one thread, in the output image");
        po::variables_map vm;
        auto po_alloptions = po::options_description()
                .add(po_generic).add(po_main);
        po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << po_alloptions << "\n";
            return 0;
        }

        param.mode = silver_bullets::enum_item_value<Mode>(modeString);
        run(param);

        return EXIT_SUCCESS;
    }
    catch(exception& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}
