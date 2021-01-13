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

namespace s3dmm {

// Returns interval number, iinterval (0<=iinterval<=levelCount),
// such that v is between levels[iinterval-1] and levels[iinterval];
// for iinterval=0, v is less than levels[0]
// Note: v must not be NaN.
__device__ inline int findInterval(float v, const float *levels, unsigned int levelCount)
{
    if (v < levels[0])
        return 0;
    if (v >= levels[levelCount-1])
        return static_cast<int>(levelCount);
    auto i1 = 0;
    auto i2 = static_cast<int>(levelCount);
    while (i2 - i1 > 1) {
        auto i = (i1 + i2) >> 1;
        if (v < levels[i])
            i2 = i;
        else
            i1 = i;
    }
    return i2;
}

} // s3dmm
