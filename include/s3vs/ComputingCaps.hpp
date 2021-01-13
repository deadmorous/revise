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

#include "def_prop_class.hpp"
#include "silver_bullets/sync/SyncAccessor.hpp"

#include <mutex>

namespace s3vs
{

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithCompNodeCount, unsigned int, unsigned int,
                              compNodeCount, setCompNodeCount,
                              onCompNodeCountChanged, offCompNodeCountChanged);

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithGPUPerNodeCount, unsigned int, unsigned int,
                              GPUPerNodeCount, setGPUPerNodeNodeCount,
                              onGPUPerNodeCountChanged, offGPUPerNodeCountChanged);

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithCPUPerNodeCount, unsigned int, unsigned int,
                              CPUPerNodeCount, setCPUPerNodeNodeCount,
                              onCPUPerNodeCountChanged, offCPUPerNodeCountChanged);

S3DMM_DEF_NOTIFIED_PROP_CLASS(WithWorkerThreadPerNodeCount, unsigned int, unsigned int,
                              workerThreadPerNodeCount, setWorkerThreadPerNodeNodeCount,
                              onWorkerThreadPerNodeCountChanged, offWorkerThreadPerNodeCountChanged);

class ComputingCaps :
    public WithCompNodeCount,
    public WithGPUPerNodeCount,
    public WithCPUPerNodeCount,
    public WithWorkerThreadPerNodeCount
{
public:
    ComputingCaps() :
        WithCompNodeCount(1),
        WithGPUPerNodeCount(1),
        WithCPUPerNodeCount(1),
        WithWorkerThreadPerNodeCount(0) // 0 means that thread count will be set equal to gpu count
    {}
};

using SyncComputingCaps = silver_bullets::sync::SyncAccessor<ComputingCaps, std::mutex>;

} // namespace s3vs
