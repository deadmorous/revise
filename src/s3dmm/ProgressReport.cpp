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

#ifdef S3DMM_PROGRESS_REPORT

#include "s3dmm/ProgressReport.hpp"

#include <boost/timer/timer.hpp>
#include <boost/assert.hpp>
#include <vector>
#include <iostream>
#include <sstream>

namespace s3dmm {

namespace {

inline ProgressReport::time_type toSeconds(boost::timer::nanosecond_type ns) {
    return static_cast<ProgressReport::time_type>(ns) / 1e9;
}

inline ProgressReport::times toProgressReportTimes(boost::timer::cpu_times t) {
    return { toSeconds(t.wall), toSeconds(t.user), toSeconds(t.system) };
}

} // anonymous namespace

class ProgressReport::Impl
{
public:
    void startStage(const std::string& title) {
        if (m_displayReport)
            std::cout << padding() << "=== Starting stage: " << title << std::endl;
        m_stages.emplace_back(StageData{title, std::make_shared<boost::timer::cpu_timer>()});
    }

    times endStage() {
        BOOST_ASSERT(!m_stages.empty());
        auto& stageData = m_stages.back();
        auto title = stageData.title;
        auto t = stageData.timer->elapsed();
        m_stages.pop_back();
        if (m_displayReport)
            std::cout << padding() << "=== Stage finished: [" << title << "]" << boost::timer::format(t) << std::endl;
        return toProgressReportTimes(t);
    }

    bool displayReport() const {
        return m_displayReport;
    }

    void setDisplayReport(bool displayReport) {
        m_displayReport = displayReport;
    }

private:
    struct StageData {
        std::string title;
        std::shared_ptr<boost::timer::cpu_timer> timer;
    };
    std::vector<StageData> m_stages;
    bool m_displayReport = true;
    std::string padding() const {
        return std::string(4*m_stages.size(), ' ');
    }
};



std::string ProgressReport::times::format() const
{
    std::ostringstream oss;
    oss << wall << "s wall, "
        << user << "s user, "
        << system << "s system";
    return oss.str();
}



ProgressReport::ProgressReport() :
    m_impl(std::make_unique<Impl>())
{}

ProgressReport::~ProgressReport() {}

void ProgressReport::startStage(const std::string& title) {
    m_impl->startStage(title);
}

auto ProgressReport::createTimer() -> Timer*
{
    return reinterpret_cast<Timer*>(new boost::timer::cpu_timer);
}

void ProgressReport::deleteTimer(Timer *timer)
{
    delete reinterpret_cast<boost::timer::cpu_timer*>(timer);
}

void ProgressReport::startTimer(Timer *timer) {
    reinterpret_cast<boost::timer::cpu_timer*>(timer)->start();
}

auto ProgressReport::stopTimer(Timer *timer) -> times
{
    auto t = reinterpret_cast<boost::timer::cpu_timer*>(timer);
    auto dt = t->elapsed();
    t->stop();
    return toProgressReportTimes(dt);
}

ProgressReport::times ProgressReport::endStage() {
    return m_impl->endStage();
}

bool ProgressReport::displayReport() const {
    return m_impl->displayReport();
}
void ProgressReport::setDisplayReport(bool displayReport) {
    m_impl->setDisplayReport(displayReport);
}

ProgressReport& ProgressReport::instance() {
    static ProgressReport instance;
    return instance;
}

} // s3dmm

#endif // S3DMM_PROGRESS_REPORT
