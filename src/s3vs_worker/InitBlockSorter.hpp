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

#include "BlockSorter.hpp"

#include "s3vs/types.hpp"


// TODO: Remove
namespace s3vs
{

inline void initBlockSorter(
    s3dmm::BlockSorter& sorter,
    const Matrix4r& m,
    const s3dmm::BlockSorter::BBox& indexBox)
{
    vl::Vector3<s3dmm::real_type> eye, at, up, right;
    // TODO: I am not sure that 'at' is defined correctly!
    m.getAsLookAt(eye, at, up, right);
    sorter.setEye({eye[0], eye[1], eye[2]});
    sorter.setCenter({at[0], at[1], at[2]});
    auto indexBox2 = indexBox;
    auto v = indexBox.max();
    v += 1;
    indexBox2 << v;
    sorter.setBoundingBox(indexBox2);
}

} // s3vs namespace
