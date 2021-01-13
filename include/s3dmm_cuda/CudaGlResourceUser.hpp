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

#include "CudaGlResource.hpp"

#include <boost/assert.hpp>

namespace s3dmm {

namespace detail {

struct CudaGlTextureRW
{
    static void registerGlResource(gpu_ll::CudaGlResource& resource, GLuint glResource)
    {
        resource.registerTexture(glResource, gpu_ll::CudaGlResource::None);
    }
};

struct CudaGlTextureW
{
    static void registerGlResource(gpu_ll::CudaGlResource& resource, GLuint glResource)
    {
        resource.registerTexture(glResource, gpu_ll::CudaGlResource::WriteDiscard);
    }
};

struct CudaGlTextureS
{
    static void registerGlResource(gpu_ll::CudaGlResource& resource, GLuint glResource)
    {
        resource.registerTexture(glResource, gpu_ll::CudaGlResource::SurfaceLoadStore);
    }
};

} // detail

template<class DataType, class Registrator, bool AlwaysUnmap = true>
class CudaGlResourceUser
{
public:
    explicit CudaGlResourceUser(gpu_ll::CudaGlResource& resource) :
        m_resource(resource)
    {
        BOOST_ASSERT(m_resource.isRegistered());
    }

    CudaGlResourceUser(gpu_ll::CudaGlResource& resource, GLuint glResource) :
        m_resource(resource)
    {
        Registrator::registerGlResource(resource, glResource);
    }

    ~CudaGlResourceUser()
    {
        if (AlwaysUnmap && m_resource.isMapped())
            m_resource.unmap();
    }

    DataType *data()
    {
        if (!m_resource.isMapped())
            m_resource.map();
        return reinterpret_cast<DataType*>(m_resource.mappedPtr());
    }

    cudaSurfaceObject_t surface() {
        return m_resource.surface();
    }

private:
    gpu_ll::CudaGlResource& m_resource;
};

template <class DataType, bool AlwaysUnmap = true>
using CudaGlTextureRWUser = CudaGlResourceUser<DataType, detail::CudaGlTextureRW, AlwaysUnmap>;

template <class DataType, bool AlwaysUnmap = true>
using CudaGlTextureWUser = CudaGlResourceUser<DataType, detail::CudaGlTextureW, AlwaysUnmap>;

template <class DataType, bool AlwaysUnmap = true>
using CudaGlTextureSUser = CudaGlResourceUser<DataType, detail::CudaGlTextureS, AlwaysUnmap>;

} // s3dmm
