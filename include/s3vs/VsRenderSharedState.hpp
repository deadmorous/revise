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

#include "VsControllerInput.hpp"

#include "s3dmm/BlockTreeFieldService.hpp"

namespace s3vs
{

struct VsRenderSharedState
{
    // Only set once, before any thread starts doing anything
    std::string shaderPath;

    // Changes rarely by setting new instance
    std::shared_ptr<s3dmm::BlockTreeFieldService<3>> fieldSvc;

    // Changes frequently
    VsControllerInput input;

    // Data transformed from input
    unsigned int primaryFieldIndex = 0;
    unsigned int secondaryFieldIndex = 0;

    // Changes frequently
    Matrix4r cameraTransform;
};

} // namespace s3vs
