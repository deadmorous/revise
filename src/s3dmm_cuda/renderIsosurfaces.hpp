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

#include "cuda_lighting.hpp"
#include "cuda_argb_transform.hpp"
#include "cuda_blend.hpp"
#include "cuda_findInterval.hpp"
#include "RenderFuncInput.hpp"
#include "RenderKernelParam.hpp"
#include "RenderFuncCaller.hpp"

#include "s3vs/FieldMode.hpp"

namespace s3dmm {

__device__ inline std::uint32_t renderIsosurfaces(const RenderFuncInput& in, const RenderKernelParam& p)
{
    auto levelCount = p.presentation.levelCount;
    if (levelCount < 1)
        return 0;

    auto& field = p.primaryField;
    auto v1 = in.fieldValue(field.tex);
    increase(in.pos, in.step);

    // Find interval in levels that contain v1 (see findInterval())
    // Special value -1 means that the field is not defined (its value is NaN).
    auto iinterval = isnan(v1)? -1: findInterval(v1, p.presentation.levels, levelCount);

    float4 rgba = { 0, 0, 0, 0 };
    for (int i=1; i<in.sampleCount; ++i, increase(in.pos, in.step)) {
        auto v2 = in.fieldValue(field.tex);
        if (isnan(v2)) {
            iinterval = -1;
            v1 = NAN;
            continue;
        }
        if (isnan(v1)) {
            v1 = v2;
            iinterval = findInterval(v1, p.presentation.levels, levelCount);
            continue;
        }
        int ilevel;
        int ilevelDelta;
        if (v2 < v1) {
            if (iinterval == 0) {
                v1 = v2;
                continue;
            }
            ilevel = iinterval - 1;
            ilevelDelta = -1;
        }
        else {
            if (iinterval == levelCount) {
                v1 = v2;
                continue;
            }
            ilevel = iinterval;
            ilevelDelta = 1;
        }
        while (ilevel >= 0 && ilevel < levelCount) {
            auto level = p.presentation.levels[ilevel];
            if ((v1-level)*(v2-level) < 0) {
                auto v01 = (level - field.minValue) / (field.maxValue - field.minValue);
                auto diffuse = tex1D<float4>(p.presentation.colorTex, v01);
                diffuse.w *= p.presentation.isosurfaceOpacity;
                auto t = (level - v1) / (v2 - v1);
                auto xpos = in.pos + in.step*t;
                auto color = computeFragColor(
                    field.tex,
                    diffuse,
                    p.eye.pos,
                    xpos,
                    (xpos - in.cubeOrigin)*in.texCoordFactor,
                    p.lights);
                blendBehind(rgba, color);
                iinterval += ilevelDelta;
                ilevel += ilevelDelta;
            }
            else
                break;
        }
        constexpr auto AlphaThreshold = 0.996f;
        if (rgba.w > AlphaThreshold)
            break;
        v1 = v2;
    }
    auto existingRgba = uint32ArgbToFloatRgba(p.viewport.pixels[in.pixelIndex]);
    blendAbove(existingRgba, rgba);
    return floatRgbaToUint32Argb(existingRgba);
}

template <>
struct RenderFuncCaller<s3vs::FieldMode::Isosurfaces>
{
    __device__ static std::uint32_t call(const RenderFuncInput& in, const RenderKernelParam& p) {
        return renderIsosurfaces(in, p);
    }
};

} // s3dmm
