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

#include "silver_bullets/factory.hpp"
#include "s3vs/VsWorkerInterface.hpp"
#include "s3dmm_cuda/DenseFieldRenderer_cu.hpp"
#include "BlockSorter.hpp"

namespace s3vs
{

class VsWorker_cu : public VsWorkerInterface,
                 public silver_bullets::FactoryMixin<VsWorker_cu, VsWorkerInterface>
{
public:
    ~VsWorker_cu();

    void initialize(
            const std::string& shaderPath,
            bool logInfo,
            const std::string& logFileName,
            size_t threadId) override;

    void setRenderSharedState(const VsRenderSharedState* sharedState) override;
    const VsRenderSharedState* renderSharedState() const override;

    void updateState(unsigned int flags) override;

    RgbaImagePart renderScenePart(
            const SubtreeSetDescription& subtrees,
            const silver_bullets::sync::CancelController::Checker& isCancelled) override;

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    const std::vector<VsWorkerTimestamps>& timestamps() const override;
    void clearTimestamps() override;
    void enableNvProf() override;
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

private:
    const VsRenderSharedState* m_sharedState = nullptr;
    s3dmm::DenseFieldRenderer_cu m_fieldRenderer;
    s3dmm::DeviceVector<std::uint32_t> m_viewport;
    std::vector<std::uint32_t> m_viewportHostBuf;
    s3dmm::BlockSorter m_blockSorter;
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    std::vector<VsWorkerTimestamps> m_timestamps;
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

    void updateProblemPath();
    void updateViewportSize();
    void updateBackgroundColor();
    void updateField();
    void updateTimeFrame();
    void updateFieldMode();
    void updateCameraPosition();
    void updateColorTransfer();
    void updateFieldParam();
    void updateClippingPlanes();
    void updateCameraFovY();
    void updateRenderPatience();
    void updateRenderQuality();
    void updateRenderLevel();
};

} // namespace s3vs
