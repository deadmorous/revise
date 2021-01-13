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

#include "Vec.hpp"
#include "ScalarOrMultiIndex.hpp"

namespace s3dmm {

template <unsigned int N, class T>
class BoundingBox
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, T>;

    bool empty() const {
        return m_empty;
    }

    void clear() {
        m_empty = true;
    }

    const vector_type& min() const {
        BOOST_ASSERT(!m_empty);
        return m_min;
    }

    const vector_type& max() const {
        BOOST_ASSERT(!m_empty);
        return m_max;
    }

    BoundingBox& operator<<(const vector_type& x)
    {
        if (m_empty) {
            m_min = m_max = x;
            m_empty = false;
        }
        else {
            for (auto i=0u; i<N; ++i) {
                auto& mini = ScalarOrMultiIndex<N, T>::element(m_min, i);
                auto& maxi = ScalarOrMultiIndex<N, T>::element(m_max, i);
                auto& xi = ScalarOrMultiIndex<N, T>::element(x, i);
                if (mini > xi)
                    mini = xi;
                else if (maxi < xi)
                    maxi = xi;
            }
        }
        return *this;
    }

    vector_type size() const {
        return m_max - m_min;
    }

    Vec<2, T> range(unsigned int dim) const {
        BOOST_ASSERT(dim < N);
        return {m_min[dim], m_max[dim]};
    }

    bool intersects(const BoundingBox<N, T>& that) const
    {
        for (auto i=0u; i<N; ++i) {
            auto& xmin = ScalarOrMultiIndex<N,T>::element(m_min, i);
            auto& xmax = ScalarOrMultiIndex<N,T>::element(m_max, i);
            auto& xminOther = ScalarOrMultiIndex<N,T>::element(that.m_min, i);
            auto& xmaxOther = ScalarOrMultiIndex<N,T>::element(that.m_max, i);
            if (xmaxOther < xmin || xmax < xminOther)
                return false;
        }
        return true;
    }

    bool contains(const vector_type& x) const
    {
        for (auto i=0u; i<N; ++i) {
            auto& xi = ScalarOrMultiIndex<N,T>::element(x, i);
            auto& xmin = ScalarOrMultiIndex<N,T>::element(m_min, i);
            auto& xmax = ScalarOrMultiIndex<N,T>::element(m_max, i);
            if (xi < xmin || xmax < xi)
                return false;
        }
        return true;
    }

    bool contains(const BoundingBox<N,T>& that) const {
        return contains(that.min()) && contains(that.max());
    }

    BoundingBox<N,T> intersection(const BoundingBox<N,T>& that) const
    {
        BoundingBox<N, T> result;
        for (auto i=0u; i<N; ++i) {
            auto& xmin = ScalarOrMultiIndex<N,T>::element(m_min, i);
            auto& xmax = ScalarOrMultiIndex<N,T>::element(m_max, i);
            auto& xminOther = ScalarOrMultiIndex<N,T>::element(that.m_min, i);
            auto& xmaxOther = ScalarOrMultiIndex<N,T>::element(that.m_max, i);
            if (xmaxOther < xmin || xmax < xminOther)
                return result;
            ScalarOrMultiIndex<N,T>::element(result.m_min, i) = std::max(xmin, xminOther);
            ScalarOrMultiIndex<N,T>::element(result.m_max, i) = std::min(xmax, xmaxOther);
        }
        result.m_empty = false;
        return result;
    }

    BoundingBox<N,T> operator&(const BoundingBox<N,T>& that) const {
        return intersection(that);
    }

    BoundingBox<N,T> offset(const vector_type& dx) const
    {
        BoundingBox<N,T> result;
        if (!m_empty) {
            result.m_empty = false;
            result.m_min = m_min + dx;
            result.m_max = m_max + dx;
        }
        return result;
    }

private:
    bool m_empty = true;
    vector_type m_min;
    vector_type m_max;
};

} // s3dmm
