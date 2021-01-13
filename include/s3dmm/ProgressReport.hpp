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

#pragma once

#ifdef S3DMM_PROGRESS_REPORT
#include <memory>
#include <string>
#include <boost/noncopyable.hpp>
#include "defs.h"
#else // S3DMM_PROGRESS_REPORT
#include <boost/noncopyable.hpp>
#endif // S3DMM_PROGRESS_REPORT

#ifdef S3DMM_PROGRESS_REPORT

namespace s3dmm {

class S3DMM_API ProgressReport : boost::noncopyable
{
public:
    using time_type = double;   // Time in seconds
    struct times {
        time_type wall = 0;
        time_type user = 0;
        time_type system = 0;
        times& operator+=(const times& that) {
            wall += that.wall;
            user += that.user;
            system += that.system;
            return *this;
        }
        template<class X>
        times& operator*=(X x) {
            wall *= x;
            user *= x;
            system *= x;
            return *this;
        }
        template<class X>
        times& operator/=(X x) {
            wall /= x;
            user /= x;
            system /= x;
            return *this;
        }
        times operator+(const times& that) const {
            times result = *this;
            result += that;
            return result;
        }
        template<class X>
        times operator*(X x) const {
            times result = *this;
            result *= x;
            return result;
        }
        template<class X>
        times operator/(X x) const {
            times result = *this;
            result /= x;
            return result;
        }
        std::string format() const;
    };
    void startStage(const std::string& title);

    // Returns stage duration
    times endStage();

    class Timer;
    static Timer *createTimer();
    static void deleteTimer(Timer *timer);
    static void startTimer(Timer *timer);
    static times stopTimer(Timer *timer);

    struct TimerDeleter {
        void operator()(Timer *timer) {
            if (timer)
                ProgressReport::deleteTimer(timer);
        }
    };

    bool displayReport() const;
    void setDisplayReport(bool displayReport);

    static ProgressReport& instance();
private:
    ProgressReport();
    ~ProgressReport();
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

class ScopedProgressReportStage : boost::noncopyable
{
public:
    ScopedProgressReportStage() : m_active(false) {}
    explicit ScopedProgressReportStage(const std::string& title) : m_active(true) {
        ProgressReport::instance().startStage(title);
    }
    // Returns duration of previous stage, or all-zeros if there was no previous stage
    ProgressReport::times startStage(const std::string& title) {
        auto result = endStage();
        ProgressReport::instance().startStage(title);
        m_active = true;
        return result;
    }
    // Returns stage duration
    ProgressReport::times endStage() {
        if (m_active) {
            m_active = false;
            return ProgressReport::instance().endStage();
        }
        else
            return ProgressReport::times{0,0,0};
    }
    ~ScopedProgressReportStage() {
        endStage();
    }
private:
    bool m_active;
};

class ScopedTimer : boost::noncopyable
{
public:
    ScopedTimer() : m_timer(ProgressReport::createTimer()) {}
    ~ScopedTimer() {
        ProgressReport::deleteTimer(m_timer);
    }
    ProgressReport::Timer *timer() const {
        return m_timer;
    }
    void addTime(const ProgressReport::times& dt) {
        m_totalTime += dt;
    }
    const ProgressReport::times& totalTime() const {
        return m_totalTime;
    }
private:
    ProgressReport::Timer *m_timer;
    ProgressReport::times m_totalTime;
};

class ScopedTimerUser : boost::noncopyable
{
public:
    explicit ScopedTimerUser(ScopedTimer *scopedTimer) :
        m_scopedTimer(scopedTimer)
    {
        start();
    }

    ~ScopedTimerUser() {
        stop();
    }

    void stop() {
        if (m_scopedTimer) {
            m_scopedTimer->addTime(ProgressReport::stopTimer(m_scopedTimer->timer()));
            m_scopedTimer = nullptr;
        }
    }

    void replace(ScopedTimer *scopedTimer)
    {
        stop();
        m_scopedTimer = scopedTimer;
        start();
    }

private:
    ScopedTimer *m_scopedTimer;

    void start()
    {
        if (m_scopedTimer)
            ProgressReport::startTimer(m_scopedTimer->timer());
    }
};

} // s3dmm

#define REPORT_PROGRESS_START_STAGE(name, title) s3dmm::ScopedProgressReportStage sprs_##name(title)
#define REPORT_PROGRESS_END_STAGE(name) sprs_##name.endStage()

#define REPORT_PROGRESS_STAGES() s3dmm::ScopedProgressReportStage sprs
#define REPORT_PROGRESS_STAGE(title) sprs.startStage(title)
#define REPORT_PROGRESS_END() sprs.endStage()
#define REPORT_DISPLAY_ENABLE(enable) s3dmm::ProgressReport::instance().setDisplayReport(enable)
#define REPORT_DISPLAY_IS_ENABLED() s3dmm::ProgressReport::instance().displayReport()
void setDisplayReport(bool displayReport);

#else // S3DMM_PROGRESS_REPORT

class ProgressReport : boost::noncopyable
{
public:
    using time_type = double;   // Time in seconds
    struct times {
        time_type wall = 0;
        time_type user = 0;
        time_type system = 0;
        times& operator+=(const times& that) {
            wall += that.wall;
            user += that.user;
            system += that.system;
            return *this;
        }
    };
private:
    ProgressReport() = default;
};


#define REPORT_PROGRESS_START_STAGE(name, title)
#define REPORT_PROGRESS_END_STAGE(name) ProgressReport::times()

#define REPORT_PROGRESS_STAGES()
#define REPORT_PROGRESS_STAGE(title) ProgressReport::times()
#define REPORT_PROGRESS_END() ProgressReport::times()
#define REPORT_DISPLAY_ENABLE(enable)
#define REPORT_DISPLAY_IS_ENABLED() false
#endif // S3DMM_PROGRESS_REPORT
