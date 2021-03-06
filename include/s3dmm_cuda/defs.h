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

#ifdef __CUDACC__
#define S3DMM_CUDA_HOST __host__
#define S3DMM_CUDA_DEVICE __device__
#define S3DMM_CUDA_HOST_AND_DEVICE __host__ __device__
#else // __CUDACC__
#define S3DMM_CUDA_HOST
#define S3DMM_CUDA_DEVICE
#define S3DMM_CUDA_HOST_AND_DEVICE
#endif // __CUDACC__

#if defined(_WIN32) && defined(_MSC_VER)
#ifdef s3dmm_cuda_EXPORTS
#define S3DMM_CUDA_CLASS_API __declspec(dllexport)
#else // s3dmm_cuda_EXPORTS
#define S3DMM_CUDA_CLASS_API __declspec(dllimport)
#endif // s3dmm_cuda_EXPORTS
#else // defined(_WIN32) && defined(_MSC_VER)
#define S3DMM_CUDA_CLASS_API
#endif // defined(_WIN32) && defined(_MSC_VER)

#define S3DMM_CUDA_API S3DMM_CUDA_CLASS_API S3DMM_CUDA_HOST
