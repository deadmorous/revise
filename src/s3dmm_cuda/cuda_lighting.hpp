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

#include "cuda_vec_op.hpp"

namespace s3dmm {

// Computes a simplified lighting equation
__device__ float3 blinn(
    const float3& N,
    const float3& V,
    const float3& L,
    const LightSource& light,
    const float3& diffuse)
{
    // material properties
    // you might want to put this into a bunch or uniforms
    float3 Ka = { 1.0, 1.0, 1.0 };
    float3 Kd = diffuse; // { 1.0, 1.0, 1.0 };
    float3 Ks = { 0.5, 0.5, 0.5 };
    float shininess = 100.0;

    // diffuse coefficient
    float diff_coeff = max( dot( L, N ), 0.0 );

    // specular coefficient
    float3 H = normalize( L + V );
    float spec_coeff = diff_coeff > 0.0 ? pow( max( dot( H, N ), 0.0 ), shininess ) : 0.0;

    // final lighting model
    return Ka * light.ambient +
           Kd * light.diffuse  * diff_coeff +
           Ks * light.specular * spec_coeff;
}

__device__ float4 computeFragColor(
    cudaTextureObject_t fieldTex,
    const float4& diffuse,
    const float3& eyePos,
    const float3& surfPos,
    const float3& texCoord,
    const RenderKernelLightsParam& lights)
{
    // compute lighting at isosurface point

    // compute the gradient and lighting only if the pixel is visible "enough"
    float3 N;
    // on-the-fly gradient computation: slower than texture but requires less memory (no gradient texture required).
    float3 a, b;
    constexpr auto gradient_delta = 0.1f;
    a.x = tex3D<float>(fieldTex, texCoord.x - gradient_delta, texCoord.y, texCoord.z);
    b.x = tex3D<float>(fieldTex, texCoord.x + gradient_delta, texCoord.y, texCoord.z);
    a.y = tex3D<float>(fieldTex, texCoord.x, texCoord.y - gradient_delta, texCoord.z);
    b.y = tex3D<float>(fieldTex, texCoord.x, texCoord.y + gradient_delta, texCoord.z);
    a.z = tex3D<float>(fieldTex, texCoord.x, texCoord.y, texCoord.z - gradient_delta);
    b.z = tex3D<float>(fieldTex, texCoord.x, texCoord.y, texCoord.z + gradient_delta);
    N  = normalize( a - b );

    float3 V  = normalize( eyePos - surfPos );
    float3 final_color = { 0.f, 0.f, 0.f };
    for(unsigned int i=0; i<lights.sourceCount; ++i)
    {
        auto& lightSource = lights.sources[i];
        float3 L = normalize(lightSource.pos - surfPos);
        // double sided lighting
        if (dot(L, N) < 0.f)
            N = -N;
        final_color = final_color + blinn( N, V, L, lightSource, {diffuse.x, diffuse.y, diffuse.z});
    }

    return {
        clamp01(final_color.x),
        clamp01(final_color.y),
        clamp01(final_color.z),
        diffuse.w
    };
}

} // s3dmm
