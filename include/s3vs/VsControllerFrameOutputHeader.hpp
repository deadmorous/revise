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

#include <cstdint>

namespace s3vs
{

/// @brief Data header that is stored in shared memory shmem of VsControllerFrameOutput
///
/// Use VsControllerFrameOutputReader and VsControllerFrameOutputWriter for i/o to shared memory
struct VsControllerFrameOutputHeader
{
    using FrameNumberType = uint64_t;

    struct FrameRenderingDuration
    {
        /// @brief Duration of rendering of the frame stored in milliseconds
        uint64_t durationMs{0};
        /// @brief Detalization level of the frame stored. ~0 if invalid.
        unsigned int level{0};
    };

    /// @brief The number of the frame stored in the shmem, or ~0 if there is no frame stored
    FrameNumberType frameNumber;

    /// @brief An information about rendering duration of the frame stored.
    ///
    /// The value is meaningful only when VsControllerOpts::measureRenderingTime option is
    /// turned on
    FrameRenderingDuration renderingDuration;
};

} // namespace s3vs
