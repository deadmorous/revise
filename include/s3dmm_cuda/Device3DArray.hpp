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
#include "Vec.hpp"
#include "CudaChannelFormat.hpp"

#include <boost/noncopyable.hpp>

struct cudaArray;
typedef unsigned long long cudaSurfaceObject_t;

namespace s3dmm {

class S3DMM_CUDA_CLASS_API Device3DArray : boost::noncopyable
{
public:
    Device3DArray() = default;
    S3DMM_CUDA_HOST ~Device3DArray();

    // Returns true if really resized
    S3DMM_CUDA_HOST bool resize(
            std::size_t width, std::size_t height, std::size_t depth,
            const CudaChannelFormat& desc);

    S3DMM_CUDA_HOST Vec3<std::size_t> extent() const {
        return m_extent;
    }

    S3DMM_CUDA_HOST void free();

    S3DMM_CUDA_HOST cudaArray *handle() const {
        return m_a;
    }

    S3DMM_CUDA_HOST CudaChannelFormat desc() const {
        return m_desc;
    }

    S3DMM_CUDA_HOST cudaSurfaceObject_t surface() const;

    S3DMM_CUDA_HOST void upload(const void *data);
    S3DMM_CUDA_HOST void download(void *data) const;

private:
    cudaArray *m_a = nullptr;
    CudaChannelFormat m_desc;
    Vec3<std::size_t> m_extent;
    unsigned int m_bytesPerElement = 0;
    mutable cudaSurfaceObject_t m_surf = 0;
};

} // s3dmm
