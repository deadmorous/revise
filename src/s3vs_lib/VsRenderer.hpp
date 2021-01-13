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

#include "IndexCubeSplitStorage.hpp"
#include "RenderGraphGeneratorPool.hpp"

#include "s3vs/VsControllerInterface.hpp"
#include "s3vs/VsControllerFrameOutputHeader.hpp"
#include "s3vs/VsWorkerTimestamps.hpp"

#include "silver_bullets/task_engine.hpp"

#include <thread>

namespace s3vs
{

namespace task_engine = silver_bullets::task_engine;

struct VsRenderSharedStateEx;

struct WorkerTimeInfo;

class VsRenderer : boost::noncopyable
{
public:
    explicit VsRenderer(const VsControllerInput *input,
                        ComputingCaps& compuCaps,
                        VsControllerOpts& opts);
    ~VsRenderer();

    void setShaderPath(const std::string& shaderPath);
    std::string shaderPath() const;

    void start();

    void setCameraTransform(const Matrix4r& cameraTransform);
    Matrix4r cameraTransform() const;

    std::vector<std::string> fieldNames() const;
    Vec2r fieldRange(const std::string& fieldName) const;
    unsigned int timeFrameCount() const;

    VsControllerFrameOutput frameOutput() const;

    void onProblemPathChanged();
    void onTimeFrameChanged();
    void onPrimaryFieldChanged();
    void onFieldModeChanged();
    void onClippingPlanesChanged();
    void onViewportSizeChanged();
    void onBackgroundColorChanged();
    void onFovYChanged();
    void onRenderPatienceChanged();
    void onRenderQualityChanged();
    void onRenderLevelChanged();
    void onFieldThresholdChanged();
    void onFieldColorTransferFunctionChanged();
    void onFieldIsosurfaceLevelChanged();
    void onFieldIsosurfaceLevelsChanged();
    void onFieldIsosurfaceOpacityChanged();
    void onFieldSecondaryFieldChanged();

private:
    const VsControllerInput *m_input;

    bool m_vsWorkersInitialized = false;

    using TaskFunc = task_engine::StatefulCancellableTaskFunc;
    using TFR = task_engine::TaskFuncRegistry<TaskFunc>;
    using TTX = task_engine::ThreadedTaskExecutor<TaskFunc>;
    using TGX = task_engine::TaskGraphExecutor<TaskFunc>;

    TFR m_taskFuncRegistry;

    // Note: must be declared before m_executor and m_mainExecutor
    silver_bullets::sync::CancelController m_cancelController;

    // Note: must be declared before m_drawWorkers, since those use shared state.
    boost::any m_drawWorkersSharedState;
    std::mutex m_renderSharedStateMutex;

    std::vector<std::shared_ptr<TTX>> m_drawWorkers;
    std::vector<std::shared_ptr<TTX>> m_assembleWorkers;
    TGX m_executor;

    VsRenderSharedStateEx& renderSharedState();
    const VsRenderSharedStateEx& renderSharedState() const;

    void maybeInitWorkers();

    // Stuff for main render thread
    using MainTaskFunc = task_engine::TaskQueueFunc;
    using MainTFR = task_engine::TaskFuncRegistry<MainTaskFunc>;
    using MainTTX = task_engine::ThreadedTaskExecutor<MainTaskFunc>;
    MainTFR m_mainTaskFuncRegistry;
    task_engine::TaskQueueExecutor m_mainExecutor;

    mutable VsControllerFrameOutput m_frameOutput;
    mutable std::mutex m_frameOutputMutex;

    const ComputingCaps& m_compuCaps;
    const VsControllerOpts& m_opts;
    IndexCubeSplitStorage m_indexCubeSplitStorage;
    RenderGraphGeneratorPool m_renderGraphGenerators;

    VsControllerFrameOutputHeader::FrameNumberType m_frameNumber{0};
    RgbaImage m_composedImage;

    std::vector<std::shared_ptr<WorkerTimeInfo>> m_workerTimeInfo;

    void updateRenderLocalState() const;
    void continueRendering(const silver_bullets::sync::CancelController::Checker& isCancelled);

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    std::vector<const std::vector<VsWorkerTimestamps>*> m_vsWorkerTimestamps;
    void clearRenderTimestamps() const;

    bool m_firstFrameRendered = false;
    void enableNvprof() const;
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

    template<class Upd>
    void updateRenderSharedState(const Upd& upd);

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    void writeWorkerTimeData(
        uint64_t time,
        const std::chrono::steady_clock::time_point& curTime,
        unsigned int level,
        bool isCancelled) const;
    void writeRenderWorkerTimestamps() const;
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
    VsControllerFrameOutputHeader::FrameNumberType m_frameNumberSaving{0};
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
};

} // namespace s3vs
