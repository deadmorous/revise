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

#include "defs.h"
#include "Vec.hpp"

#include "s3dmm_cuda/CudaGlResource.hpp"
#include "s3dmm_cuda/Device3DArray.hpp"

#include <GL/gl.h>

namespace s3dmm {

class S3DMM_CUDA_CLASS_API DenseFieldRenderHelperArr_cu
{
public:
    static void prepareTextures(
            unsigned int depth,
            const Device3DArray& fieldDev,
            gpu_ll::CudaGlResource& fieldTex,
            gpu_ll::CudaGlResource& alphaTex,
            const Vec2<dfield_real>& dfieldRange);
};

} // s3dmm
