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

#include <boost/noncopyable.hpp>

namespace s3dmm {

class Device3DArray;
class DeviceArray;
// template <class T> class DeviceVector;

class DeviceTextureObject : boost::noncopyable
{
public:
    DeviceTextureObject() = default;
    __host__ ~DeviceTextureObject();

    __host__ void createBoundTexture(
        const Device3DArray& array,
        bool normalizedCoords,
        cudaTextureFilterMode filterMode,
        cudaTextureAddressMode addressMode);

    __host__ void createBoundTexture(
        const DeviceArray& array,
        bool normalizedCoords,
        cudaTextureFilterMode filterMode,
        cudaTextureAddressMode addressMode);
    /*
    __host__ void createBoundTexture(
        const DeviceVector<unsigned char>& v,
        bool normalizedCoords,
        cudaTextureFilterMode filterMode,
        cudaTextureAddressMode addressMode);

    __host__ void createBoundTexture(
        const DeviceVector<float>& v,
        bool normalizedCoords,
        cudaTextureFilterMode filterMode,
        cudaTextureAddressMode addressMode);
*/
    __host__ void free();

    __host__ cudaTextureObject_t handle() const {
        return m_t;
    }

private:
    cudaTextureObject_t m_t = 0;
};

} // s3dmm

