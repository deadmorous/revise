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

#include <cstdint>

namespace s3dmm {

__device__ inline float4 uint32ArgbToFloatRgba(std::uint32_t argb)
{
    return {
        ((argb >> 16) & 0xffu) / 255.f,
        ((argb >> 8 ) & 0xffu) / 255.f,
        ( argb        & 0xffu) / 255.f,
        ((argb >> 24) & 0xffu) / 255.f
    };
}

__device__ inline std::uint32_t floatRgbaToUint32Argb(const float4& rgba)
{
    return
        (static_cast<unsigned int>(rgba.w * 255.999f) << 24) +
        (static_cast<unsigned int>(rgba.x * 255.999f) << 16) +
        (static_cast<unsigned int>(rgba.y * 255.999f) << 8) +
        static_cast<unsigned int>(rgba.z * 255.999f);
}

} // s3dmm
