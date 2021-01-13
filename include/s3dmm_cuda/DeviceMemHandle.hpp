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

#include "defs.h"

#include <cstddef>

#include <boost/noncopyable.hpp>

namespace s3dmm {

namespace gpu_ll {

class S3DMM_CUDA_CLASS_API DeviceMemHandle : boost::noncopyable
{
public:
    DeviceMemHandle() = default;
    S3DMM_CUDA_HOST DeviceMemHandle(DeviceMemHandle&& that);
    S3DMM_CUDA_HOST explicit DeviceMemHandle(std::size_t byteCount);
    S3DMM_CUDA_HOST ~DeviceMemHandle();

    S3DMM_CUDA_HOST DeviceMemHandle& operator=(DeviceMemHandle&& that);

    S3DMM_CUDA_HOST void resize(std::size_t byteCount);
    S3DMM_CUDA_HOST void free();

    S3DMM_CUDA_HOST std::size_t byteCount() const;
    S3DMM_CUDA_HOST_AND_DEVICE void *data() const;
    S3DMM_CUDA_HOST void clear() const;
    S3DMM_CUDA_HOST void upload(const void *src) const;
    S3DMM_CUDA_HOST void upload(const void *src, std::size_t byteCount, std::size_t dstStartIndex) const;
    S3DMM_CUDA_HOST void download(void *dst) const;
    S3DMM_CUDA_HOST void download(void *dst, std::size_t byteCount, std::size_t srcStartIndex) const;

private:
    void *m_d = nullptr;
    std::size_t m_byteCount = 0;
    std::size_t m_allocatedByteCount = 0;
};

} // gpu_ll

} // s3dmm
