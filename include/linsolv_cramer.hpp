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

#include "real_type.hpp"

namespace s3dmm {
namespace linsolv_crammer {

inline void invert2x2(real_type *dst, const real_type *src)
{
    auto iDelta = make_real(1) / (src[0]*src[3] - src[1]*src[2]);
    dst[0] =  iDelta*src[3];
    dst[1] = -iDelta*src[1];
    dst[2] = -iDelta*src[2];
    dst[3] =  iDelta*src[0];
}

inline void solve2x2(real_type *result, const real_type *mx, const real_type *rhs)
{
    auto iDelta = make_real(1) / (mx[0]*mx[3] - mx[1]*mx[2]);
    result[0] = (rhs[0]*mx[3] - rhs[1]*mx[2]) * iDelta;
    result[1] = (mx[0]*rhs[1] - mx[1]*rhs[0]) * iDelta;
}

inline void invert3x3(real_type *dst, const real_type *src)
{
    auto d11 = src[4]*src[8]-src[5]*src[7];
    auto d21 = src[3]*src[8]-src[5]*src[6];
    auto d31 = src[3]*src[7]-src[4]*src[6];
    auto d12 = src[1]*src[8]-src[2]*src[7];
    auto d22 = src[0]*src[8]-src[2]*src[6];
    auto d32 = src[0]*src[7]-src[1]*src[6];
    auto d13 = src[1]*src[5]-src[2]*src[4];
    auto d23 = src[0]*src[5]-src[2]*src[3];
    auto d33 = src[0]*src[4]-src[1]*src[3];
    auto id = make_real(1) / (src[0]*d11 - src[1]*d21 + src[2]*d31);
    dst[0] =  id*d11;
    dst[1] = -id*d12;
    dst[2] =  id*d13;
    dst[3] = -id*d21;
    dst[4] =  id*d22;
    dst[5] = -id*d23;
    dst[6] =  id*d31;
    dst[7] = -id*d32;
    dst[8] =  id*d33;
}

inline void solve3x3(real_type *result, const real_type *mx, const real_type *rhs)
{
    auto d11 = mx[4]*mx[8]-mx[5]*mx[7];
    auto d21 = mx[3]*mx[8]-mx[5]*mx[6];
    auto d31 = mx[3]*mx[7]-mx[4]*mx[6];
    auto id = make_real(1) / (mx[0]*d11 - mx[1]*d21 + mx[2]*d31);
    auto d1 =  rhs[0]*d11 - rhs[1]*d21 + rhs[2]*d31;
    auto d2 = -rhs[0]*(mx[1]*mx[8]-mx[2]*mx[7]) + rhs[1]*(mx[0]*mx[8]-mx[2]*mx[6]) - rhs[2]*(mx[0]*mx[7]-mx[1]*mx[6]);
    auto d3 =  rhs[0]*(mx[1]*mx[5]-mx[2]*mx[4]) - rhs[1]*(mx[0]*mx[5]-mx[2]*mx[3]) + rhs[2]*(mx[0]*mx[4]-mx[1]*mx[3]);
    result[0] = id*d1;
    result[1] = id*d2;
    result[2] = id*d3;
}

} // linsolv_crammer
} // s3dmm
