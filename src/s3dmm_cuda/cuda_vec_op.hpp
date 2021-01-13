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

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return { a.x*s, a.y*s, a.z*s };
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return { a.x*b.x, a.y*b.y, a.z*b.z };
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return { -a.x, -a.y, -a.z };
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline float3 normalize(const float3& a)
{
    float il = 1.f / sqrt(dot(a, a));
    return { a.x*il, a.y*il, a.z*il };
}

__host__ __device__ inline void increase(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ inline float3& asFloat3(float4& x) {
    return reinterpret_cast<float3&>(x);
}

__host__ __device__ inline const float3& asFloat3(const float4& x) {
    return reinterpret_cast<const float3&>(x);
}

} // s3dmm
