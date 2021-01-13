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

#include "BoundingBox.hpp"
#include "s3dmm/TreeBlockId.hpp"

#include <vector>

#include <boost/range/iterator_range.hpp>

#include <vlCore/Matrix4.hpp>

namespace s3vs
{

using Block3Id = s3dmm::detail::TreeBlockId<3>;
// TODO: Remove using Block3IdRange = boost::iterator_range<const Block3Id*>;

using Vec2i = s3dmm::Vec<2, int>;
using Rect2i = s3dmm::BoundingBox<2, int>;

using Vec2r = s3dmm::Vec<2, s3dmm::real_type>;
using Vec3r = s3dmm::Vec<3, s3dmm::real_type>;
using Vec4r = s3dmm::Vec<4, s3dmm::real_type>;

using Matrix3r = vl::Matrix3<s3dmm::real_type>;
using Matrix4r = vl::Matrix4<s3dmm::real_type>;

struct RgbaImage
{
    Vec2i size;
    std::vector<unsigned char> bits;
};

struct RgbaImagePart
{
    Vec2i origin;
    RgbaImage image;
};

} // namespace s3vs
