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

#ifdef HIRES_TIME_USE_RDTSC

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>

namespace s3dmm {
using hires_time_t = __int64;
} // s3dmm

#endif // _WIN32

#ifdef __linux__

#include <x86intrin.h>

namespace s3dmm {
using hires_time_t = unsigned long long;
} // s3dmm

#endif // __linux__

namespace s3dmm {

inline hires_time_t hires_time() {
    return __rdtsc();
}

} // s3dmm
#endif // HIRES_TIME_USE_RDTSC


#ifdef _WIN32
#error "TODO: hires_time.h on Windows"
#endif // _WIN32

#ifdef __linux__

#include <time.h>
#include <stdexcept>
#include <ostream>
#include <iomanip>

namespace s3dmm {

struct hires_time_t : timespec {
    hires_time_t(int = 0) {
        tv_sec = 0;
        tv_nsec = 0;
    }
};

inline std::ostream& operator<<(std::ostream& s, const hires_time_t& t)
{
    s << std::setprecision(14);
    auto sec = static_cast<double>(t.tv_sec) + 1e-9*t.tv_nsec;
    s << sec;
    return s;
}

inline hires_time_t hires_time()
{
    static hires_time_t initTime;
    static bool initTimeComputed = false;
    hires_time_t result;
    if (clock_gettime(CLOCK_MONOTONIC, &result) != 0)
        throw std::runtime_error("hires_time() is not working due to clock_gettime()");
    if (!initTimeComputed) {
        initTimeComputed = true;
        initTime = result;
    }
    result.tv_sec -= initTime.tv_sec;
    return result;
}

} // s3dmm

#endif // __linux__
