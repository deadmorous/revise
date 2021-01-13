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

#include "VsRenderSharedState.hpp"
#include "SubtreeSetDescription.hpp"
#include "VsWorkerTimestamps.hpp"

#include "silver_bullets/factory.hpp"
#include "silver_bullets/sync/CancelController.hpp"

namespace s3dmm
{
template <unsigned int N>
class BlockTreeFieldService;
}

namespace s3vs
{

struct VsWorkerInterface : public silver_bullets::Factory<VsWorkerInterface>
{
    /// @brief Flags passed in a bitwise-or combination to renderScenePart().
    enum StateUpdateFlag
    {
        UpdateProblemPath       = 0x0001,
        UpdateViewportSize      = 0x0002,
        UpdateBackgroundColor   = 0x0004,
        UpdateField             = 0x0008,
        UpdateTimeFrame         = 0x0010,
        UpdateFieldMode         = 0x0020,
        UpdateCameraPosition    = 0x0040,
        UpdateColorTransfer     = 0x0080,
        UpdateFieldParam        = 0x0100,
        UpdateClippingPlanes    = 0x0200,
        UpdateCameraFovY        = 0x0400,
        UpdateRenderPatience    = 0x0800,
        UpdateRenderQuality     = 0x1000,
        UpdateRenderLevel       = 0x2000
    };

    virtual ~VsWorkerInterface() = default;

    /**
     * @brief initialization provides base initialization of vl library, opengl
     * and so on.
     * @param shaderPath path to shaders
     */
    virtual void initialize(
            const std::string& shaderPath,
            bool logInfo,
            const std::string& logFileName,
            size_t threadId) = 0;

    /**
     * @brief Sets render shared state
     */
    virtual void setRenderSharedState(const VsRenderSharedState* sharedState) = 0;

    /**
     * @brief Returns render shared state previously set using setRenderSharedState()
     */
    virtual const VsRenderSharedState* renderSharedState() const = 0;

    /**
     * @brief Updates internal state to reflect changes of VsRenderSharedState.
     * @param flags A combination of elements of StateUpdateFlag enum.
     */
    virtual void updateState(unsigned int flags) = 0;

    /**
     * @brief renderScenePart Visualizes specific field subtrees on the scene.
     * @param sceneState Current state of the scene.
     * @param subtreeRoots Field subtrees to be rendered.
     * @return Framebuffer pixels, cropped to the minimal area
     * containing all pixels with field from visualized subtrees.
     */
    virtual RgbaImagePart renderScenePart(
            const SubtreeSetDescription& subtrees,
            const silver_bullets::sync::CancelController::Checker& isCancelled) = 0;

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    /**
     * @brief Returns timestamps of operations performed during calls to renderScenePart() done since last call to clearTimestamps().
     */
    virtual const std::vector<VsWorkerTimestamps>& timestamps() const = 0;

    /**
     * @brief Clears timestamps returned by the timestamps() method.
     */
    virtual void clearTimestamps() = 0;
    virtual void enableNvProf() = 0;
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
};

} // namespace s3vs
