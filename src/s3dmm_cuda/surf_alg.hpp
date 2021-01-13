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

#include "cudaCheck.hpp"
#include "computeBlockCount.hpp"

template<class Tin, class Tout, class F>
__global__ void transformKernel(
    cudaSurfaceObject_t surfIn,
    cudaSurfaceObject_t surfOut,
    unsigned int verticesPerEdge, F f)
{
    auto ix = threadIdx.x + blockDim.x*blockIdx.x;
    auto iy = threadIdx.y + blockDim.y*blockIdx.y;
    auto iz = threadIdx.z + blockDim.z*blockIdx.z;
    if (ix < verticesPerEdge && iy < verticesPerEdge && iz < verticesPerEdge) {
        auto v = surf3Dread<Tin>(surfIn, ix*sizeof(Tin), iy, iz);
        surf3Dwrite<Tout>(f(v), surfOut, ix*sizeof(Tout), iy, iz);
    }
}

template<
    class Tin, class Tout,
    class WithSurfIn, class WithSurfOut,
    class F>
inline void transform(
        unsigned int verticesPerEdge,
        WithSurfIn& ain,
        WithSurfOut& aout,
        F f)
{
    constexpr unsigned int blockSize1d = 8;
    auto blockCount1d = computeBlockCount(verticesPerEdge, blockSize1d);
    dim3 blockSize(blockSize1d, blockSize1d, blockSize1d);
    dim3 blockCount(blockCount1d, blockCount1d, blockCount1d);
    transformKernel<Tin, Tout, F><<<blockCount, blockSize>>>(
        ain.surface(),
        aout.surface(),
        verticesPerEdge, f);
    CU_CHECK(cudaPeekAtLastError());
}



template<class T, class F>
__global__ void transformInPlaceKernel(
    cudaSurfaceObject_t surf,
    unsigned int verticesPerEdge, F f)
{
    auto ix = threadIdx.x + blockDim.x*blockIdx.x;
    auto iy = threadIdx.y + blockDim.y*blockIdx.y;
    auto iz = threadIdx.z + blockDim.z*blockIdx.z;
    if (ix < verticesPerEdge && iy < verticesPerEdge && iz < verticesPerEdge) {
        auto v = surf3Dread<T>(surf, ix*sizeof(T), iy, iz);
        surf3Dwrite<T>(f(v), surf, ix*sizeof(T), iy, iz);
    }
}

template<class T, class WithSurf, class F>
inline void transformInPlace(
        unsigned int verticesPerEdge,
        WithSurf& a,
        F f)
{
    constexpr unsigned int blockSize1d = 8;
    auto blockCount1d = computeBlockCount(verticesPerEdge, blockSize1d);
    dim3 blockSize(blockSize1d, blockSize1d, blockSize1d);
    dim3 blockCount(blockCount1d, blockCount1d, blockCount1d);
    transformInPlaceKernel<T, F><<<blockCount, blockSize>>>(
        a.surface(), verticesPerEdge, f);
    CU_CHECK(cudaPeekAtLastError());
}



template<class T>
__global__ void fillKernel(
    cudaSurfaceObject_t surf,
    unsigned int verticesPerEdge, T value)
{
    auto ix = threadIdx.x + blockDim.x*blockIdx.x;
    auto iy = threadIdx.y + blockDim.y*blockIdx.y;
    auto iz = threadIdx.z + blockDim.z*blockIdx.z;
    if (ix < verticesPerEdge && iy < verticesPerEdge && iz < verticesPerEdge)
        surf3Dwrite<T>(value, surf, ix*sizeof(T), iy, iz);
}

template<class T, class WithSurf>
inline void fill(
        unsigned int verticesPerEdge,
        WithSurf& a,
        T value)
{
    constexpr unsigned int blockSize1d = 8;
    auto blockCount1d = computeBlockCount(verticesPerEdge, blockSize1d);
    dim3 blockSize(blockSize1d, blockSize1d, blockSize1d);
    dim3 blockCount(blockCount1d, blockCount1d, blockCount1d);
    fillKernel<T><<<blockCount, blockSize>>>(a.surface(), verticesPerEdge, value);
    CU_CHECK(cudaPeekAtLastError());
}
