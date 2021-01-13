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

struct cudaChannelFormatDesc;

namespace s3dmm {

class CudaChannelFormat
{
public:
    enum Kind {
        Signed = 0,
        Unsigned = 1,
        Float = 2,
        None = 3
    };
    unsigned int x = 0;
    unsigned int y = 0;
    unsigned int z = 0;
    unsigned int w = 0;
    Kind f = Signed;

    S3DMM_CUDA_CLASS_API cudaChannelFormatDesc makeCudaChannelFormatDesc() const;
    static S3DMM_CUDA_CLASS_API cudaChannelFormatDesc makeCudaChannelFormatDesc(
        const CudaChannelFormat& channelFormat);

    bool operator==(const CudaChannelFormat& that) const {
        return x == that.x && y == that.y && z == that.z && w == that.w && f == that.f;
    }
    bool operator!=(const CudaChannelFormat& that) const {
        return !(*this == that);
    }
};

} // s3dmm
