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

struct RenderFuncInput
{
    unsigned int vx;
    unsigned int vy;
    float x;
    float y;
    float3 ray;
    float tauEnter;
    float tauExit;
    unsigned int pixelIndex;
    unsigned int n;
    float texelSize;
    int sampleCount;
    float3 step;
    mutable float3 pos;
    float3 cubeOrigin;
    float texCoordFactor;

    __device__ float fieldValue(const cudaTextureObject_t& tex) const {
        return fieldValue(tex, pos);
    }

    __device__ float fieldValue(const cudaTextureObject_t& tex, const float3& x) const {
        auto texCoord = (x - cubeOrigin)*texCoordFactor;
        return tex3D<float>(tex, texCoord.x, texCoord.y, texCoord.z);
    }
};

} // s3dmm

