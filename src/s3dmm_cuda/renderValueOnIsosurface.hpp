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
#include "RenderFuncInput.hpp"
#include "RenderKernelParam.hpp"
#include "RenderFuncCaller.hpp"

#include "s3vs/FieldMode.hpp"

namespace s3dmm {

__device__ inline std::uint32_t renderValueOnIsosurface(const RenderFuncInput& in, const RenderKernelParam& p)
{
    auto& pfield = p.primaryField;
    auto& sfield = p.secondaryField;
    auto v1 = in.fieldValue(pfield.tex);
    increase(in.pos, in.step);

    float4 rgba = { 0, 0, 0, 0 };
    for (int i=1; i<in.sampleCount; ++i, increase(in.pos, in.step)) {
        auto v2 = in.fieldValue(pfield.tex);
        if (!isnan(v1) && !isnan(v2) && (v1-p.presentation.threshold)*(v2-p.presentation.threshold) < 0) {
            auto t = (p.presentation.threshold - v1) / (v2 - v1);
            auto xpos = in.pos + in.step*t;
            auto vsec = in.fieldValue(sfield.tex, xpos);
            auto vsec01 = (vsec - sfield.minValue) / (sfield.maxValue - sfield.minValue);
            auto diffuse = tex1D<float4>(p.presentation.colorTex, vsec01);
            diffuse.w *= p.presentation.isosurfaceOpacity;
            auto color = computeFragColor(
                pfield.tex,
                diffuse,
                p.eye.pos,
                xpos,
                (xpos - in.cubeOrigin)*in.texCoordFactor,
                p.lights);
            blendBehind(rgba, color);
            constexpr auto AlphaThreshold = 0.996f;
            if (rgba.w > AlphaThreshold)
                break;
        }
        v1 = v2;
    }
    auto existingRgba = uint32ArgbToFloatRgba(p.viewport.pixels[in.pixelIndex]);
    blendAbove(existingRgba, rgba);
    return floatRgbaToUint32Argb(existingRgba);
}

template <>
struct RenderFuncCaller<s3vs::FieldMode::ValueOnIsosurface>
{
    __device__ static std::uint32_t call(const RenderFuncInput& in, const RenderKernelParam& p) {
        return renderValueOnIsosurface(in, p);
    }
};

} // s3dmm
