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

#include "ws_sendframe/FrameServerThread.hpp"
#include "ws_sendframe/FrameServer.hpp"

#include <QThread>

class FrameServerThread::Impl : public QThread
{
public:
    explicit Impl(unsigned int port) : m_port(port) {}
    FrameServer *frameServer() const {
        return m_frameServer;
    }

protected:
    void run() override {
        FrameServer frameServer(m_port);
        m_frameServer = &frameServer;
        exec();
        m_frameServer = nullptr;
    }

private:
    unsigned int m_port;
    FrameServer *m_frameServer = nullptr;
};

FrameServerThread::FrameServerThread(unsigned short port) :
    m_impl(new Impl(port))
{
    QObject::connect(m_impl, &QThread::finished, m_impl, &QObject::deleteLater);
    m_impl->start();
}

FrameServerThread::~FrameServerThread() {
    m_impl->exit(0);
}

FrameServer *FrameServerThread::frameServer() const {
    return m_impl->frameServer();
}
