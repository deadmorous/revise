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
#include "s3vs/types.hpp"

#include <boost/assert.hpp>

#include <algorithm>

namespace s3vs
{


inline s3dmm::BoundingBox<2, int>& operator<<(
    s3dmm::BoundingBox<2, int>& bb, const RgbaImage& img)
{
    bb << Vec2i{0, 0} << img.size;
    return bb;
}

inline s3dmm::BoundingBox<2, int> imageBoundingBox(const RgbaImage& img)
{
    s3dmm::BoundingBox<2, int> bb;
    bb << img;
    return bb;
}

inline s3dmm::BoundingBox<2, int>& operator<<(
    s3dmm::BoundingBox<2, int>& bb, const RgbaImagePart& imgpt)
{
    bb << imgpt.origin << (imgpt.origin + imgpt.image.size);
    return bb;
}

inline s3dmm::BoundingBox<2, int> imagePartBoundingBox(const RgbaImagePart& imgpt)
{
    s3dmm::BoundingBox<2, int> bb;
    bb << imgpt;
    return bb;
}

void blendImagePart(
    RgbaImage& dst,
    const RgbaImagePart& src,
    bool justCopySrc = false,
    const Vec2i& dstOffset = Vec2i{0, 0})
{
    BOOST_ASSERT(imageBoundingBox(dst).offset(dstOffset).contains(
        imagePartBoundingBox(src)));
    auto spix = reinterpret_cast<const uint32_t*>(src.image.bits.data());
    auto sw = src.image.size[0];
    auto sh = src.image.size[1];
    auto dpix = reinterpret_cast<uint32_t*>(dst.bits.data());
    auto dw = dst.size[0];
    //auto spixRow = spix;
    auto dx = src.origin[0] - dstOffset[0];
    auto dy = src.origin[1] - dstOffset[1];

    auto getByte = [](const uint32_t& val, const uint32_t& n){
        return (val >> (n << 3)) & 0xFF;
    };

    const uint32_t byteMask[] = {
        (~0U) ^ 255U,
        (~0U) ^ (255U << 8),
        (~0U) ^ (255U << 16),
        (~0U) ^ (255U << 24)
    };

    auto setByte = [byteMask](uint32_t& dst, const uint32_t& n, const uint32_t& src){
        dst &= byteMask[n];
        dst |= ((src & 255U) << (n << 3));
    };

    if (justCopySrc)
    {
#pragma omp parallel for
        for (auto sy = 0; sy < sh; ++sy)
        {
            auto spixRow = spix + sy*sw;
            auto dpixRow = dpix + dw * (dy + sy) + dx;
            std::copy(spixRow, spixRow + sw, dpixRow);
        }
        return;
    }

#pragma omp parallel for
    for (auto sy = 0; sy < sh; ++sy)
    {
        auto spixRow = spix + sy*sw;
        auto dpixRow = dpix + dw * (dy + sy) + dx;
        for (auto i = 0; i < sw; ++i)
        {
            const auto& s = spixRow[i];
            auto& d = dpixRow[i];

            auto as = getByte(s, 3);

            if (as == 0)
                continue; // fully transparent foreground

            if (as == 255U) // fully opaque foreground
            {
                d = s;
                continue;
            }

            auto ad = getByte(d, 3);

            if (ad == 0) // fully transparent background
            {
                d = s;
                continue;
            }

            auto a = as + ad*(255U - as)/255U;
            setByte(d, 3, a);

            for (auto i = 0; i < 3; i++)
            {
                auto c = (getByte(s, i)*as + getByte(d, i)*(a - as))/a;
                setByte(d, i, c);
            }
        }
    }
}


void blendImagePartMaxIntensity(
    RgbaImage& dst,
    const RgbaImagePart& src,
    bool justCopySrc = false,
    const Vec2i& dstOffset = Vec2i{0, 0})
{
    BOOST_ASSERT(imageBoundingBox(dst).offset(dstOffset).contains(
        imagePartBoundingBox(src)));
    auto spix = reinterpret_cast<const float*>(src.image.bits.data());
    auto sw = src.image.size[0];
    auto sh = src.image.size[1];
    auto dpix = reinterpret_cast<float*>(dst.bits.data());
    auto dw = dst.size[0];
    auto dx = src.origin[0] - dstOffset[0];
    auto dy = src.origin[1] - dstOffset[1];

    if (justCopySrc)
    {
#pragma omp parallel for
        for (auto sy = 0; sy < sh; ++sy)
        {
            auto spixRow = spix + sy*sw;
            auto dpixRow = dpix + dw * (dy + sy) + dx;
            std::copy(spixRow, spixRow + sw, dpixRow);
        }
        return;
    }

#pragma omp parallel for
    for (auto sy = 0; sy < sh; ++sy)
    {
        auto spixRow = spix + sy*sw;
        auto dpixRow = dpix + dw * (dy + sy) + dx;
        for (auto i = 0; i < sw; ++i)
        {
            const auto& s = spixRow[i];
            if (std::isnan(s))
                continue;
            auto& d = dpixRow[i];
            if (std::isnan(d) || d < s)
                d = s;
        }
    }
}

} // namespace s3vs
