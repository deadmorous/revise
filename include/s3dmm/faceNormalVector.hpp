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

namespace s3dmm {

inline Vec<3, real_type> faceNormalVector(const MultiIndex<3, unsigned int>& face, const Vec<3, real_type> *nodes)
{
    auto& r0 = nodes[face[0]];
    auto& r1 = nodes[face[1]];
    auto& r2 = nodes[face[2]];
    auto n = (r1 - r0) % (r2 - r0);
    return n / norm(n);
}

inline Vec<3, real_type> faceNormalVector(const MultiIndex<4, unsigned int>& face, const Vec<3, real_type> *nodes)
{
    auto& r0 = nodes[face[0]];
    auto& r1 = nodes[face[1]];
    auto& r2 = nodes[face[2]];
    auto& r3 = nodes[face[3]];
    auto r01 = r1 - r0;
    auto r12 = r2 - r1;
    auto r23 = r3 - r2;
    auto r30 = r0 - r3;
    auto n = r01%r12 + r12%r23 + r23%r30 + r30%r01;
    auto nn = norm(n);
    if (nn > 0)
        return n / nn;
    else
        // Note: zero normal vector means degenerate face!
        return Vec<3, real_type>();
}

inline Vec<2, real_type> faceNormalVector(const MultiIndex<2, unsigned int>& face, const Vec<2, real_type> *nodes)
{
    auto& r0 = nodes[face[0]];
    auto& r1 = nodes[face[1]];
    auto n = rot_cw(r1 - r0);
    auto nn = norm(n);
    if (nn > 0)
        return n / nn;
    else
        // Note: zero normal vector means degenerate face!
        return Vec<2, real_type>();
}

inline Vec<1, real_type> faceNormalVector(const MultiIndex<1, unsigned int>& /*face*/, const Vec<1, real_type>* /*nodes*/) {
    return { make_real(1) };
}

} // s3dmm
