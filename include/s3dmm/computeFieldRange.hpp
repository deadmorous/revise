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

#include <Vec.hpp>

namespace s3dmm {

template <class T>
inline Vec2<T> computeFieldRange(const std::vector<T>& fieldValues, const T& noFieldValue)
{
    auto hasMinMax = false;
    s3dmm::Vec<2, T> minMax = { 0, 0 };
    for (auto v : fieldValues) {
        if (v != noFieldValue) {
            if (hasMinMax) {
                if (minMax[0] > v)
                    minMax[0] = v;
                else if (minMax[1] < v)
                    minMax[1] = v;
            }
            else {
                minMax[0] = minMax[1] = v;
                hasMinMax = true;
            }
        }
    }
    return hasMinMax? minMax: Vec2<T>{ T(0), T(-1) };
}

} // s3dmm
