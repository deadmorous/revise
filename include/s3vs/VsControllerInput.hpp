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

#include "types.hpp"
#include "def_prop_class.hpp"
#include "silver_bullets/sync/SyncAccessor.hpp"
#include "FieldMode.hpp"

#include <mutex>

/// @brief S3DMM visualization server namespace.
namespace s3vs
{

/// @brief Field visualization mode property.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithFieldMode, FieldMode, FieldMode,
    fieldMode, setFieldMode, onFieldModeChanged, offFieldModeChanged);

/// @brief Base class for the level of single isosurface visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithIsosurfaceLevelBase, s3dmm::real_type, const s3dmm::real_type&,
    isosurfaceLevel, setIsosurfaceLevel, onIsosurfaceLevelChanged, offIsosurfaceLevelChanged);

/// @brief The dimensionless level of single isosurface visualized, from 0 to 1.
class WithIsosurfaceLevel :
        public WithIsosurfaceLevelBase
{
public:
    /// @brief Initializes level with the value of 0.5
    WithIsosurfaceLevel() : WithIsosurfaceLevelBase(s3dmm::make_real(0.5)) {}
};

/// @brief The levels of multiple isosurfaces visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithIsosurfaceLevels, std::vector<s3dmm::real_type>, const std::vector<s3dmm::real_type>&,
    isosurfaceLevels, setIsosurfaceLevels, onIsosurfaceLevelsChanged, offIsosurfaceLevelsChanged);

/// @brief The opacity of isosurface(s) visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithIsosurfaceOpacityBase, s3dmm::real_type, const s3dmm::real_type&,
    isosurfaceOpacity, setIsosurfaceOpacity, onIsosurfaceOpacityChanged, offIsosurfaceOpacityChanged);

/// @brief Base class for the opacity of isosurface(s) visualized.
class WithIsosurfaceOpacity :
        public WithIsosurfaceOpacityBase
{
public:
    /// @brief Initializes opacity with the value of 1
    WithIsosurfaceOpacity() : WithIsosurfaceOpacityBase(s3dmm::make_real(1)) {}
};

/// @brief The name of secondary field to be visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithSecondaryField, std::string, const std::string&,
    secondaryField, setSecondaryField, onSecondaryFieldChanged, offSecondaryFieldChanged);

/// @brief Base class for the threshold value to be used for visualization.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithThresholdBase, s3dmm::real_type, const s3dmm::real_type&,
    threshold, setThreshold, onThresholdChanged, offThresholdChanged);

/// @brief The threshold value to be used for visualization.
class WithThreshold :
        public WithThresholdBase
{
public:
    /// @brief Initializes threshold with the value of 0.5
    WithThreshold() : WithThresholdBase(s3dmm::make_real(0.5)) {}
};

using ColorTransferFunction = std::map<s3dmm::real_type, Vec4r>;

/// @brief The transfer function to be used for visualization.
///
/// The transfer function is used to map value in the range [0, 1] to RGBA color.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithColorTransferFunction, ColorTransferFunction, const ColorTransferFunction&,
    colorTransferFunction, setColorTransferFunction,
    onColorTransferFunctionChanged, offColorTransferFunctionChanged);

/// @brief Field parameters for specific field visualization mode.
template <FieldMode fieldMode> struct VsFieldParam;

/// @brief Field parameters for non-isosurface field visualization modes.
struct VsFieldNonIsosurfaceParam :
        virtual WithThreshold,
        virtual WithColorTransferFunction
{};

template<> struct VsFieldParam<FieldMode::Isosurface> :
        virtual WithIsosurfaceLevel,
        virtual WithIsosurfaceOpacity,
        virtual WithColorTransferFunction
{};

template<> struct VsFieldParam<FieldMode::Isosurfaces> :
        virtual WithIsosurfaceLevels,
        virtual WithIsosurfaceOpacity,
        virtual WithColorTransferFunction
{};
template<> struct VsFieldParam<FieldMode::MaxIntensityProjection> : VsFieldNonIsosurfaceParam {};
template<> struct VsFieldParam<FieldMode::Argb> : VsFieldNonIsosurfaceParam {};
template<> struct VsFieldParam<FieldMode::ArgbLight> : VsFieldNonIsosurfaceParam {};
template<> struct VsFieldParam<FieldMode::DomainVoxels> : VsFieldNonIsosurfaceParam {};

template<> struct VsFieldParam<FieldMode::ValueOnIsosurface> :
        virtual WithIsosurfaceLevel,
        virtual WithIsosurfaceOpacity,
        virtual WithColorTransferFunction,
        virtual WithSecondaryField
{};

template<> struct VsFieldParam<FieldMode::ValueOnIsosurfaces> :
        virtual WithIsosurfaceLevels,
        virtual WithIsosurfaceOpacity,
        virtual WithColorTransferFunction,
        virtual WithSecondaryField
{};

/// @brief All field visualization parameters.
struct VsFieldAllParam :
        VsFieldParam<FieldMode::Isosurface>,
        VsFieldParam<FieldMode::Isosurfaces>,
        VsFieldParam<FieldMode::MaxIntensityProjection>,
        VsFieldParam<FieldMode::Argb>,
        VsFieldParam<FieldMode::ArgbLight>,
        VsFieldParam<FieldMode::DomainVoxels>,
        VsFieldParam<FieldMode::ValueOnIsosurface>,
        VsFieldParam<FieldMode::ValueOnIsosurfaces>
{
};

/// @brief Path to problem whose numerical solution is to be visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithProblemPath, std::string, const std::string&,
    problemPath, setProblemPath, onProblemPathChanged, offProblemPathChanged);

/// @brief Ordinal number of time frame.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithTimeFrame, unsigned int, const unsigned int&,
    timeFrame, setTimeFrame, onTimeFrameChanged, offTimeFrameChanged);

/// @brief The name of primary field to be visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithPrimaryField, std::string, const std::string&,
    primaryField, setPrimaryField, onPrimaryFieldChanged, offPrimaryFieldChanged);

/// @brief Clipping plane to apply to field being visualized.
///
/// The part of the field lying in the semiplane on the side where the normal vector points to
/// should not be visualized.
struct ClippingPlane
{
    /// @brief A point on the clipping plane.
    Vec3r pos;

    /// @brief Normal vector to the clipping plane (must be nonzero).
    Vec3r normal;
};

/// @brief A set of clipping planes to be applied to field visualized.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithClippingPlanes, std::vector<ClippingPlane>, const std::vector<ClippingPlane>&,
    clippingPlanes, setClippingPlanes, onClippingPlanesChanged, offClippingPlanesChanged);

/// @brief The size of the wiewport for which the output is generated.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithViewportSize, Vec2i, const Vec2i&,
    viewportSize, setViewportSize, onViewportSizeChanged, offViewportSizeChanged);

/// @brief Background color.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithBackgroundColor, Vec3r, const Vec3r&,
                              backgroundColor, setBackgroundColor, onBackgroundColorChanged, offBackgroundColorChanged);

/// @brief Camera field of view in the vertical direction (the value is in degrees).
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithFovY, s3dmm::real_type, const s3dmm::real_type&,
    fovY, setFovY, onFovYChanged, offFovYChanged);

/// @brief Max. time interval between generated frames, in milliseconds.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithRenderPatience, unsigned int, const unsigned int&,
    renderPatience, setRenderPatience, onRenderPatienceChanged, offRenderPatienceChanged);

/// @brief Number of steps per ray in volume raycast algorithms.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithRenderQuality, s3dmm::real_type, const s3dmm::real_type&,
    renderQuality, setRenderQuality, onRenderQualityChanged, offRenderQualityChanged);

/// @brief Number of metadata level to render.
///
/// A non-negative value is clamped to [0, max_level], and only one metadata is rendered,
/// which disables progressive rendering.
/// A negative value enables normal progressive rendering.
S3DMM_DEF_NOTIFIED_PROP_CLASS(WithRenderLevel, int, const int&,
                              renderLevel, setRenderLevel, onRenderLevelChanged, offRenderLevelChanged);



/// @brief Visualization controller input parameters.
///
/// This class contains all parameters that define the scene,
/// except camera transformation, which is updated as user drags scene with the mouse
class VsControllerInput :
    public WithProblemPath,
    public WithTimeFrame,
    public WithPrimaryField,
    public WithFieldMode,
    public WithClippingPlanes,
    public WithViewportSize,
    public WithBackgroundColor,
    public WithFovY,
    public WithRenderPatience,
    public WithRenderQuality,
    public WithRenderLevel
{
public:
    VsControllerInput() :
        WithFieldMode(FieldMode::Isosurface),
        WithViewportSize({800, 600}),
        WithFovY(30),
        WithRenderPatience(40),
        WithRenderQuality(1),
        WithRenderLevel(-1)
    {}

    template<FieldMode fieldMode>
    VsFieldParam<fieldMode>& fieldParam() {
        return m_fieldAllParam;
    }

    template<FieldMode fieldMode>
    const VsFieldParam<fieldMode>& fieldParam() const {
        return m_fieldAllParam;
    }

    VsFieldAllParam& fieldAllParam() {
        return m_fieldAllParam;
    }

    const VsFieldAllParam& fieldAllParam() const {
        return m_fieldAllParam;
    }

private:
    VsFieldAllParam m_fieldAllParam;
};

using SyncVsControllerInput = silver_bullets::sync::SyncAccessor<VsControllerInput, std::mutex>;
using SyncVsFieldAllParam = silver_bullets::sync::SyncAccessor<VsFieldAllParam, std::mutex>;

} // namespace s3vs
