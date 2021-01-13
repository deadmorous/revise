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

namespace s3vs
{

/// @brief Field visualization mode enumeration.
enum class FieldMode
{
    /// @brief Display a single isosurface of the field.
    Isosurface,

    /// @brief Display multiple isosurfaces of the field.
    Isosurfaces,

    /// @brief Pixel is colored according to maximum value along the ray
    MaxIntensityProjection,

    /// @brief Argb volume rendering.
    Argb,

    /// @brief Argb volume rendering; each sample is shaded according to the lights.
    ArgbLight,

    /// @brief Display voxels belonging to the domain.
    DomainVoxels,

    /// @brief Display secondary field value on a single isosurface of primary field.
    ValueOnIsosurface,

    /// @brief Display secondary field value on multiple isosurfaces of primary field.
    ValueOnIsosurfaces
};

} // namespace s3vs
