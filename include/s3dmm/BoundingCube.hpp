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

#include "BoundingBox.hpp"

#define S3DMM_CENTER_BOUNDING_CUBE_AT_BBOX_CENTER

namespace s3dmm {

template <unsigned int N, class T>
class BoundingCube
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, T>;

    BoundingCube() : m_size(0) {}

    BoundingCube(const vector_type& pos, const T& size) :
        m_min(pos),
        m_size(size)
    {}

    explicit BoundingCube(const BoundingBox<N,T>& bb) :
        m_min(bb.min()),
        m_size(ScalarOrMultiIndex<N,T>::max(bb.size()))
    {
#ifdef S3DMM_CENTER_BOUNDING_CUBE_AT_BBOX_CENTER
        m_min -= m_size/2;
        m_min += bb.size()/2;
#endif // S3DMM_CENTER_BOUNDING_CUBE_AT_BBOX_CENTER
    }

    const vector_type& min() const {
        return m_min;
    }

    vector_type max() const {
        return ScalarOrMultiIndex<N,T>::transform(m_min, [this](const T& x) { return x + m_size; });
    }

    vector_type center() const {
        auto halfSize = make_real(0.5) * m_size;
        return ScalarOrMultiIndex<N,T>::transform(m_min, [halfSize](const T& x) { return x + halfSize; });
    }

    T size() const {
        return m_size;
    }

    bool contains(const vector_type& x) const {
        return contains(x, m_size, 0);
    }

    bool contains(const vector_type& x, real_type tol) const {
        return contains(x, m_size, tol);
    }

    bool contains(const BoundingCube<N,T>& bc) const {
        auto d = m_size - bc.size();
        return d > 0?  contains(bc.min(), d, 0): false;
    }

    bool contains(const BoundingCube<N,T>& bc, real_type tol) const {
        auto d = m_size - bc.size();
        return d > 0?  contains(bc.min(), d, tol): false;
    }

    bool contains(const BoundingBox<N,T>& bb) const
    {
        BOOST_ASSERT(!bb.empty());
        return contains(bb.min()) && contains(bb.max());
    }

    bool contains(const BoundingBox<N,T>& bb, real_type tol) const
    {
        BOOST_ASSERT(!bb.empty());
        return contains(bb.min(), tol) && contains(bb.max(), tol);
    }

private:
    vector_type m_min;
    T m_size;

    bool contains(const vector_type& x, real_type size, real_type tol) const
    {
        for (auto i=0u; i<N; ++i) {
            auto& xi = ScalarOrMultiIndex<N,T>::element(x, i);
            auto& xmin = ScalarOrMultiIndex<N,T>::element(m_min, i);
            if (xi < xmin-tol || xmin + size+tol < xi)
                return false;
        }
        return true;
    }
};

} // s3dmm

