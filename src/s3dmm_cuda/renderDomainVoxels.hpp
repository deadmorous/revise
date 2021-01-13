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

__device__ inline std::uint32_t renderDomainVoxels(const RenderFuncInput& in, const RenderKernelParam& p)
{
    auto& field = p.primaryField;

    auto iLayerCount = 1.f / ((1 << (field.texDepth + p.cube.level)) * p.presentation.quality);
    // This actually means that the opacity is specified for
    // a layer of thickness of 1% of total cube size
    iLayerCount *= 100.f;
    auto opacityFactor = (p.presentation.threshold - field.minValue) / (field.maxValue - field.minValue);
    float4 rgba = { 0, 0, 0, 0 };
    auto diffuse = tex1D<float4>(p.presentation.colorTex, 0.5f);
    diffuse.w = 1.f - pow(1.f - diffuse.w*opacityFactor, iLayerCount);
    for (int i=1; i<in.sampleCount; ++i, increase(in.pos, in.step)) {
        auto v = in.fieldValue(field.tex);
        if (!isnan(v)) {
            blendBehind(rgba, diffuse);
            constexpr auto AlphaThreshold = 0.996f;
            if (rgba.w > AlphaThreshold)
                break;
        }
    }
    auto existingRgba = uint32ArgbToFloatRgba(p.viewport.pixels[in.pixelIndex]);
    blendAbove(existingRgba, rgba);
    return floatRgbaToUint32Argb(existingRgba);
}

template <>
struct RenderFuncCaller<s3vs::FieldMode::DomainVoxels>
{
    __device__ static std::uint32_t call(const RenderFuncInput& in, const RenderKernelParam& p) {
        return renderDomainVoxels(in, p);
    }
};

} // s3dmm
