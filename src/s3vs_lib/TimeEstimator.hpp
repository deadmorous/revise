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

#include <boost/assert.hpp>

#include <chrono>

namespace s3vs
{

struct TimeEstimatorData
{
    double time{0.};
    size_t callCount{0};
};

class TimeEstimator
{
public:
    using clock = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<clock>;
    explicit TimeEstimator(TimeEstimatorData& data) : m_data(data)
    {
        resume();
    }
    ~TimeEstimator()
    {
        stop();
    }
private:
    void resume()
    {
        if (m_resumeCount == 0)
            m_timeCur = time_point(clock::now());
        ++m_resumeCount;
    }
    void stop()
    {
        --m_resumeCount;
        if (m_resumeCount)
            return;
        std::chrono::duration<double, std::milli> duration = clock::now() - m_timeCur;
        m_data.time = duration.count();
        m_data.callCount++;
    }

    TimeEstimatorData& m_data;
    int m_resumeCount{0};
    time_point m_timeCur;
};


#ifdef S3VS_ENABLE_TIME_ESTIMATOR
#define S3VS_TIME_ESTIMATOR(name, timeEstimatorData)\
s3vs::TimeEstimator name(timeEstimatorData);
#else
#define S3VS_TIME_ESTIMATOR(name, timeEstimatorData)
#endif

} // namespace s3vs


