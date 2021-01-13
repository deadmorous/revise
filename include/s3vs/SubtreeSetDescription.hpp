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

namespace s3vs
{

struct SubtreeSetDescription
{
    SubtreeSetDescription() = default;

    SubtreeSetDescription(
            unsigned int level,
            const s3dmm::BoundingBox<3, unsigned int>& indexBox) :
        level(level),
        indexBox(indexBox)
    {
    }

    SubtreeSetDescription(
            unsigned int level,
            const s3dmm::MultiIndex<3, unsigned int>& index) :
        level(level)
    {
        indexBox << index;
    }

    SubtreeSetDescription(
            unsigned int level,
            const s3dmm::MultiIndex<3, unsigned int>& minIndex,
            const s3dmm::MultiIndex<3, unsigned int>& maxIndex) :
        level(level)
    {
        indexBox << minIndex << maxIndex;
    }

    unsigned int level = 0;
    s3dmm::BoundingBox<3, unsigned int> indexBox;
};

} // namespace s3vs
