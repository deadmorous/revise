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

#include "DeviceMemHandle.hpp"

#include <vector>

namespace s3dmm {

template <class T>
class DeviceVector
{
public:
    using value_type = T;

    DeviceVector() = default;

    explicit DeviceVector(std::size_t count) :
        m_mh(count * sizeof(T))
    {}

    DeviceVector(DeviceVector<T>&& that) :
        m_mh(std::move(that.m_mh))
    {}

    DeviceVector(const T *src, std::size_t count) :
        m_mh(count * sizeof(T))
    {
        upload(src);
    }

    explicit DeviceVector(const std::vector<T>& src) :
        m_mh(src.size()*sizeof(T))
    {
        upload(src.data());
    }

    DeviceVector<T>& operator=(DeviceVector<T>&& that)
    {
        if (this != &that)
            m_mh = std::move(that.m_mh);
        return *this;
    }

    void resize(std::size_t count) {
        m_mh.resize(count * sizeof(T));
    }

    void free() {
        m_mh.free();
    }

    std::size_t size() const {
        return m_mh.byteCount() / sizeof (T);
    }

    T *data() {
        return reinterpret_cast<T*>(m_mh.data());
    }

    const T *data() const {
        return reinterpret_cast<T*>(m_mh.data());
    }

    T *begin() {
        return data();
    }

    const T *begin() const {
        return data();
    }

    T *end() {
        return data() + size();
    }

    const T *end() const {
        return data() + size();
    }

    void upload(const T *src) const {
        m_mh.upload(src);
    }

    void upload(const std::vector<T>& src)
    {
        resize(src.size());
        upload(src.data());
    }

    void download(T *dst) const {
        m_mh.download(dst);
    }

    void download(std::vector<T>& dst) const
    {
        dst.resize(size());
        download(dst.data());
    }

    std::vector<T> download() const
    {
        std::vector<T> result(size());
        download(result.data());
        return result;
    }

    T download(std::size_t index) const
    {
        T result;
        m_mh.download(&result, sizeof(T), index*sizeof(T));
        return result;
    }

    T upload(std::size_t index, const T& value) const {
        m_mh.upload(&value, sizeof(T), index*sizeof(T));
    }

private:
    gpu_ll::DeviceMemHandle m_mh;
};

} // s3dmm
