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

#include "RenderFuncInput.hpp"
#include "RenderKernelParam.hpp"
#include "RenderFuncCaller.hpp"

#include "s3vs/FieldMode.hpp"

namespace s3dmm {

__device__ inline std::uint32_t renderMaxIntensityProjection(const RenderFuncInput& in, const RenderKernelParam& p)
{
    auto& field = p.primaryField;
    float maxVal = field.minValue - (field.maxValue - field.minValue);
    for (int i=1; i<in.sampleCount; ++i, increase(in.pos, in.step)) {
        auto v = in.fieldValue(field.tex);
        if (v >= p.presentation.threshold && maxVal < v)
            maxVal = v;
    }

    if (maxVal < field.minValue)
        // there is no new value, so stay old (maybe nan)
        return p.viewport.pixels[in.pixelIndex];

    float t = (maxVal - field.minValue) / (field.maxValue - field.minValue);
    auto pixPrev = p.viewport.pixels[in.pixelIndex];
    float tPrev = *reinterpret_cast<const float*>(&pixPrev);
    if (!isnan(tPrev) && tPrev > t)
        return pixPrev;
    return *reinterpret_cast<std::uint32_t*>(&t);
}

template <>
struct RenderFuncCaller<s3vs::FieldMode::MaxIntensityProjection>
{
    __device__ static std::uint32_t call(const RenderFuncInput& in, const RenderKernelParam& p) {
        return renderMaxIntensityProjection(in, p);
    }
};

} // s3dmm
