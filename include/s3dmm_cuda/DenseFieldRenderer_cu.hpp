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
#include "DeviceVector.hpp"

#include "s3vs/VsRenderSharedState.hpp"
#include "s3vs/VsWorkerTimestamps.hpp"

#include <atomic>

namespace s3dmm {

class DenseFieldRenderer_cu
{
public:
    DenseFieldRenderer_cu();
    ~DenseFieldRenderer_cu();

    void setRenderSharedState(const s3vs::VsRenderSharedState* sharedState);

    const s3vs::VsRenderSharedState* renderSharedState() const;

    void clearViewport(DeviceVector<std::uint32_t>& viewport) const;

    void renderDenseField(
        DeviceVector<std::uint32_t>& viewport,
        unsigned int level,
        const s3dmm::MultiIndex<3, unsigned int>& index,
        const std::atomic<bool>& isCancelled);
    const s3vs::BlockTimestamps& timestamps() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // s3dmm
