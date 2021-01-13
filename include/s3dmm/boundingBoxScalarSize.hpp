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

namespace s3dmm {

enum class BoundingBoxScalarSizePolicy
{
    MinPositiveSize, MaxSize, AverageSize
};

namespace detail {

template<BoundingBoxScalarSizePolicy policy, unsigned int N>
struct BoundingBoxScalarSize {};

template <unsigned int N>
struct BoundingBoxScalarSize<BoundingBoxScalarSizePolicy::MinPositiveSize, N>
{
    static real_type size(const BoundingBox<N, real_type>& bb)
    {
        auto minSize = make_real(0);
        ScalarOrMultiIndex<N, real_type>::each(bb.size(), [&](real_type x) {
            if ((minSize <= 0 || minSize > x) && x > 0)
                minSize = x;
        });
        return minSize;
    }
};

template <unsigned int N>
struct BoundingBoxScalarSize<BoundingBoxScalarSizePolicy::MaxSize, N>
{
    static real_type size(const BoundingBox<N, real_type>& bb)
    {
        return ScalarOrMultiIndex<N, real_type>::max(bb.size());
    }
};

template <unsigned int N>
struct BoundingBoxScalarSize<BoundingBoxScalarSizePolicy::AverageSize, N>
{
    static real_type size(const BoundingBox<N, real_type>& bb)
    {
        auto sum = make_real(0);
        ScalarOrMultiIndex<N, real_type>::each(bb.size(), [&](real_type x) {
            sum += x;
        });
        return sum / N;
    }
};

} // detail

template<BoundingBoxScalarSizePolicy policy, unsigned int N>
inline real_type boundingBoxScalarSize(const BoundingBox<N, real_type>& bb) {
    return detail::BoundingBoxScalarSize<policy, N>::size(bb);
}

} // s3dmm
