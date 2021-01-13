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

#include <exception>
#include <string>
#include <sstream>

#include "s3dmm_cuda/defs.h"

namespace s3dmm {

namespace gpu_ll {

template<typename StatusEnum> struct StatusEnumTraits {};

template<typename StatusEnum>
class cudaExceptionTemplate : public std::exception
{
public:
    S3DMM_CUDA_HOST cudaExceptionTemplate(StatusEnum code, const char *file, unsigned int line) :
        m_code(code), m_file(file), m_line(line)
    {}
    S3DMM_CUDA_HOST const char *what() const noexcept {
        return format();
    }
private:
    StatusEnum m_code;
    const char *m_file;
    unsigned int m_line;

    mutable std::string m_buf;
    S3DMM_CUDA_HOST inline const char *format() const noexcept
    {
        using namespace std;
        using T = StatusEnumTraits<StatusEnum>;
        if (m_buf.empty()) {
            ostringstream oss;
            // oss << T::libraryName() <<  " error " << m_code << " (" << cudaDescribeError(m_code) << ")" << " in" << endl << m_file << ":" << m_line;
            oss << T::libraryName() <<  " error " << m_code << " (" << T::describeError(m_code) << ")" << " in" << endl << m_file << ":" << m_line;
            m_buf = oss.str();
        }
        return m_buf.c_str();
    }
};

template<typename StatusEnum>
S3DMM_CUDA_HOST inline void cudaCheckResult(StatusEnum code, const char *file, unsigned int line)
{
    if (code != StatusEnumTraits<StatusEnum>::Success) {
        throw cudaExceptionTemplate<StatusEnum>(code, file, line);
    }
}

} // gpu_ll

} // s3dmm
