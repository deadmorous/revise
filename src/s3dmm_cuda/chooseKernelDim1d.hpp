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

#include "computeBlockCount.hpp"

#include <utility>

namespace s3dmm {

std::pair<unsigned int, unsigned int> chooseKernelDim1d(
    unsigned int actualSize, unsigned int maxBlockSize)
{
    if (actualSize <= maxBlockSize)
        return {1, actualSize};
    else
        return {computeBlockCount(actualSize, maxBlockSize), maxBlockSize};
}

} // s3dmm
