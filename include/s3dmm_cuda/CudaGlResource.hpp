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

#include <GL/gl.h>

#include <boost/noncopyable.hpp>

typedef unsigned long long cudaSurfaceObject_t;

namespace s3dmm {
namespace gpu_ll {

class S3DMM_CUDA_CLASS_API CudaGlResource : boost::noncopyable
{
public:
    enum UsageFlag {
        None                = 0,
        ReadOnly            = 0x01,
        WriteDiscard        = 0x02,
        SurfaceLoadStore    = 0x04,
        TextureGather       = 0x08
    };
    CudaGlResource() = default;
    S3DMM_CUDA_HOST ~CudaGlResource();
    S3DMM_CUDA_HOST void registerTexture(GLuint texture, unsigned int usageFlags);
    S3DMM_CUDA_HOST void unregister();
    S3DMM_CUDA_HOST void map();
    S3DMM_CUDA_HOST void unmap();
    S3DMM_CUDA_HOST void *mappedPtr();

    GLuint glResource() const {
        return m_glResource;
    }

    unsigned int usageFlags() const {
        return m_usageFlags;
    }

    bool isRegistered() const {
        return m_glResource != 0u;
    }

    bool isMapped() const {
        return m_mapped;
    }

    S3DMM_CUDA_HOST cudaSurfaceObject_t surface();

private:
    GLuint m_glResource = 0;
    unsigned int m_usageFlags = 0;
    void *m_cudaResource = nullptr;
    bool m_mapped = false;
    cudaSurfaceObject_t m_surf = 0;
};

} // gpu_ll
} // s3dmm
