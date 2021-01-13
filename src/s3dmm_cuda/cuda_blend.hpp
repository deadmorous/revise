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

__device__ inline void blendBehind(float4& existing, const float4& incoming)
{
    auto da = incoming.w*(1.f - existing.w);
    auto a = existing.w + da;
    if (a > 0.f) {
        auto& existingColor = asFloat3(existing);
        existingColor = existingColor*(existing.w/a) + asFloat3(incoming)*(da/a);
    }
    existing.w = a;
}

__device__ inline void blendAbove(float4& existing, const float4& incoming)
{
    auto da = existing.w*(1.f - incoming.w);
    auto a = incoming.w + da;
    if (a > 0.f) {
        auto& existingColor = asFloat3(existing);
        existingColor = existingColor*(da/a) + asFloat3(incoming)*(incoming.w/a);
    }
    existing.w = a;
}

} // s3dmm
