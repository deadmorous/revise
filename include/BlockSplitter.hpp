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

namespace s3dmm
{

class BlockSplitter
{
public:
    using BBox = BoundingBox<3, unsigned int>;
    std::vector<std::vector<BBox>> split(
        unsigned int level,
        unsigned int splitCount,
        const Vec3d& dir,
        const Vec3d& e0,
        const Vec3d& e1)
    {
        std::array<Vec3d, 3> e;
        e[0] = e0 / norm(e0);
        e[2] = e0 % e1;
        e[2] = e[2] / norm(e[2]);
        e[1] = e[2] % e[0];

        std::vector<std::vector<BBox>> res(splitCount);

        // TODO

        return res;
    }
};

} // namespace s3dmm
