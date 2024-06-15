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

#include "s3vs/types.hpp"


// TODO: Remove
namespace s3vs
{

const s3dmm::BoundingBox<3, s3dmm::real_type> boundingBoxOfIndexBox(
    unsigned int level, const s3dmm::BoundingBox<3, unsigned int>& box)
{
    auto cubesPerLevel = 1 << level;
    constexpr auto TopLevelCubeHalfSize = 5.f;
    auto cubeHalfSize = TopLevelCubeHalfSize / cubesPerLevel;
    auto cubeCenterCoord = [&](unsigned int idx) {
        return -TopLevelCubeHalfSize + cubeHalfSize*(1 + (idx << 1));
    };
    auto cubeCenter = [&](const s3dmm::Vec3u& idx)
        -> s3dmm::Vec3d
    {
        return {
            cubeCenterCoord(idx[0]),
            cubeCenterCoord(idx[1]),
            cubeCenterCoord(idx[2]) };
    };
    auto d = s3dmm::Vec3d{ cubeHalfSize, cubeHalfSize, cubeHalfSize };
    auto min = cubeCenter(box.min()) - d;
    auto box_max = box.max();
    box_max -= 1;   // Because `box` contains integer `OpenRange`s.
    auto max = cubeCenter(box_max) + d;
    return s3dmm::BoundingBox<3, s3dmm::real_type>{} << min << max;
}

s3dmm::Vec3d eyeFromTransform(const Matrix4r& transform)
{
    vl::Vector3<s3dmm::real_type> eye, at, up, right;
    transform.getAsLookAt(eye, at, up, right);
    return { eye[0], eye[1], eye[2] };
}

} // s3vs namespace
