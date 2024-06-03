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

#include "VsRenderer.hpp"
#include "BlendImagePart.hpp"
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
#define S3VS_ENABLE_TIME_ESTIMATOR
#endif//S3DMM_ENABLE_WORKER_TIME_ESTIMATION
#include "TimeEstimator.hpp"
#undef S3VS_ENABLE_TIME_ESTIMATOR

#include "s3vs/ColorTransferFunction.hpp"
#include "s3vs/VsWorkerInterface.hpp"
#include "s3vs/VsControllerFrameOutputRW.hpp"

#include "s3vs/VsWorkerTimestamps.hpp"

#include "s3dmm/BlockTreeFieldService.hpp"

#include "silver_bullets/system/get_program_dir.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/range.hpp>

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
#include <QImage>
#endif //S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING

#include <chrono>
#include <filesystem>

using namespace std;
using namespace s3dmm;

namespace s3vs
{

namespace
{

template <class NameHolder>
unsigned int resolveField(
    const BlockTreeFieldService<3>& fieldService, NameHolder& h)
{
    if (fieldService.fieldCount() == 0)
        return ~0u;
    auto name = h.get();
    auto result = fieldService.maybeFieldIndex(name);
    if (result == ~0u)
    {
        h.set(fieldService.fieldName(0));
        return 0;
    }
    else
        return result;
}


using SyncVsControllerFrameOutput =
    silver_bullets::sync::SyncAccessor<VsControllerFrameOutput, mutex>;



void maybeInitFrameOutput(
    const SyncVsControllerFrameOutput& syncFrameOutput,
    const Vec2i& viewportSize)
{
    auto frameOutputAcc = syncFrameOutput.access();
    auto& frameOutput = frameOutputAcc.get();
    if (!(frameOutput.frameWidth == viewportSize[0]
          && frameOutput.frameHeight == viewportSize[1]))
    {
        frameOutput.frameWidth = viewportSize[0];
        frameOutput.frameHeight = viewportSize[1];
        frameOutput.frameSize = viewportSize[0] * viewportSize[1] * 4;
        frameOutput.shmem = SharedMemory::createNew(
            sizeof(VsControllerFrameOutputHeader) + frameOutput.frameSize);
        VsControllerFrameOutputRW writer(frameOutput);
        writer.writeFrame({
            ~0U,
            {
                0,
                ~0U
            }
        });
    }
}

inline std::uint32_t floatRgbaToUint32Argb(const Vec4r& rgba)
{
    return
        (static_cast<uint32_t>(lround(rgba[3]*255)) << 24) +
        (static_cast<uint32_t>(lround(rgba[0]*255)) << 16) +
        (static_cast<uint32_t>(lround(rgba[1]*255)) << 8) +
         static_cast<uint32_t>(lround(rgba[2]*255));
}

inline std::uint32_t floatRgbToUint32Argb(const Vec3r& rgb)
{
    return
        0xff000000 +
        (static_cast<uint32_t>(lround(rgb[0]*255)) << 16) +
        (static_cast<uint32_t>(lround(rgb[1]*255)) << 8) +
         static_cast<uint32_t>(lround(rgb[2]*255));
}

} // anonymous namespace

using namespace task_engine;

using SyncMatrix4r = silver_bullets::sync::SyncAccessor<Matrix4r, mutex>;

namespace MainTask
{
enum
{
    UpdateRenderLocalState,
    ContinueRendering,
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    ClearRenderTimestamps
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
};
} // namespace MainTask

enum class RenderStateChange
{
    ProblemPath,
    ViewportSize,
    BackgroundColor,
    Field,
    TimeFrame,
    FieldMode,
    CameraPosition,
    ColorTransfer,
    FieldParam,
    ClippingPlanes,
    CameraFovY,
    RenderPatience,
    RenderQuality,
    RenderLevel,

    Count
};

class RenderStateChanges
{
public:
    RenderStateChanges()
    {
        fill(m_timestamps.begin(), m_timestamps.end(), 0);
    }

    uint64_t get(RenderStateChange index) const
    {
        return m_timestamps.at(static_cast<size_t>(index));
    }

    void inc(RenderStateChange index)
    {
        ++m_timestamps.at(static_cast<size_t>(index));
    }

    template <class... Args>
    void inc(
        RenderStateChange index1, RenderStateChange index2, Args... moreIndices)
    {
        inc(index1);
        inc(index2, moreIndices...);
    }

    bool update(RenderStateChange index, const RenderStateChanges& source)
    {
        auto i = static_cast<size_t>(index);
        auto& dst = m_timestamps[i];
        auto& src = source.m_timestamps.at(i);
        if (dst == src)
            return false;
        else
        {
            dst = src;
            return true;
        }
    }

private:
    array<uint64_t, static_cast<size_t>(RenderStateChange::Count)> m_timestamps;
};

struct VsRenderSharedStateEx : VsRenderSharedState
{
    VsRenderSharedStateEx()
    {
        shaderPath = filesystem::path(silver_bullets::system::get_program_dir())
                     / "data";
    }

    // Timestamps of the required changes
    RenderStateChanges renderStateChanges;
};

struct WorkerTimeInfo
{
    TimeEstimatorData timeData;
    unsigned int threadId{0};
    unsigned int nodeId{0};
    unsigned int resourceType{0};
    unsigned int callCountTotal{0};
    unsigned int callCountTotalSuccess{0};
};

struct RenderThreadLocalState
{
    std::size_t threadId;
    std::size_t gpuId;
    shared_ptr<VsWorkerInterface> vsWorker;
    WorkerTimeInfo* workerTimeInfo{nullptr};

    // Timestamps of already applied changes
    RenderStateChanges renderStateChanges;
};

struct AssembleThreadLocalState
{
    std::size_t threadId;
    WorkerTimeInfo* workerTimeInfo{nullptr};
};

struct TaskAuxParam
{
    uint64_t frameNumberSaving{0};
    unsigned int taskIndex{0};
    unsigned int level{0};
};

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
namespace {

inline void saveImagePart(const RgbaImage& img, const TaskAuxParam& param, const string& taskType)
{
    if (param.level == 0U)
	return;
    string fileName = "level_" + to_string(param.level) +
                      "_frame_" + to_string(param.frameNumberSaving) +
                      "_" + taskType + "_" + to_string(param.taskIndex) + ".png";
    QImage image(img.bits.data(), img.size[0], img.size[1], QImage::Format_ARGB32);
    image.save(fileName.c_str());
}

}; // anonymous namespace
#endif //S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING

class RenderStateInitFunc : public StatefulCancellableTaskFuncInterface
{
    void call(
        boost::any& threadLocalData,
        const pany_range& /*out*/,
        const const_pany_range& /*in*/,
        const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        const override
    {
        // Init render resources
        auto& localState =
            boost::any_cast<RenderThreadLocalState&>(threadLocalData);
        auto& sharedState = boost::any_cast<const VsRenderSharedStateEx&>(
            *readOnlySharedData());
        string logFileName = "log/vl/worker_"
                             + boost::lexical_cast<string>(localState.threadId)
                             + ".log";
        localState.vsWorker->initialize(
            sharedState.shaderPath, false, logFileName, localState.gpuId);
        localState.vsWorker->setRenderSharedState(&sharedState);
    }
};

class RenderStateUpdateFunc : public StatefulCancellableTaskFuncInterface
{
    void call(
        boost::any& threadLocalData,
        const pany_range& /*out*/,
        const const_pany_range& /*in*/,
        const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        const override
    {
        auto& localState =
            boost::any_cast<RenderThreadLocalState&>(threadLocalData);
        auto& sharedState = boost::any_cast<const VsRenderSharedStateEx&>(
            *readOnlySharedData());
        auto& lu = localState.renderStateChanges;
        auto& su = sharedState.renderStateChanges;

        using WI = VsWorkerInterface;

        auto workerStateUpdates = 0u;
        if (lu.update(RenderStateChange::ProblemPath, su))
            workerStateUpdates |= WI::UpdateProblemPath;
        if (lu.update(RenderStateChange::ViewportSize, su))
            workerStateUpdates |= WI::UpdateViewportSize;
        if (lu.update(RenderStateChange::BackgroundColor, su))
            workerStateUpdates |= WI::UpdateBackgroundColor;
        if (lu.update(RenderStateChange::Field, su))
            workerStateUpdates |= WI::UpdateField;
        if (lu.update(RenderStateChange::TimeFrame, su))
            workerStateUpdates |= WI::UpdateTimeFrame;
        if (lu.update(RenderStateChange::FieldMode, su))
            workerStateUpdates |= WI::UpdateFieldMode;
        if (lu.update(RenderStateChange::CameraPosition, su))
            workerStateUpdates |= WI::UpdateCameraPosition;
        if (lu.update(RenderStateChange::ColorTransfer, su))
            workerStateUpdates |= WI::UpdateColorTransfer;
        if (lu.update(RenderStateChange::FieldParam, su))
            workerStateUpdates |= WI::UpdateFieldParam;
        if (lu.update(RenderStateChange::ClippingPlanes, su))
            workerStateUpdates |= WI::UpdateClippingPlanes;
        if (lu.update(RenderStateChange::CameraFovY, su))
            workerStateUpdates |= WI::UpdateCameraFovY;
        if (lu.update(RenderStateChange::RenderPatience, su))
            workerStateUpdates |= WI::UpdateRenderPatience;
        if (lu.update(RenderStateChange::RenderQuality, su))
            workerStateUpdates |= WI::UpdateRenderQuality;
        if (lu.update(RenderStateChange::RenderLevel, su))
            workerStateUpdates |= WI::UpdateRenderLevel;

        if (workerStateUpdates)
            localState.vsWorker->updateState(workerStateUpdates);
    }
};

class RenderFunc : public StatefulCancellableTaskFuncInterface
{
public:
    using Input = SubtreeSetDescription;

    void call(
        boost::any& threadLocalData,
        const pany_range& out,
        const const_pany_range& in,
        const silver_bullets::sync::CancelController::Checker& isCancelled)
        const override
    {
        auto& subtrees = boost::any_cast<const Input&>(*in[0]);
        auto& localState =
            boost::any_cast<RenderThreadLocalState&>(threadLocalData);
        S3VS_TIME_ESTIMATOR(estimator, localState.workerTimeInfo->timeData)
        *out[0] = localState.vsWorker->renderScenePart(subtrees, isCancelled);
#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
        auto& outImg = boost::any_cast<const RgbaImagePart&>(*out[0]);
        auto& auxParam = boost::any_cast<const TaskAuxParam&>(*in[1]);
        saveImagePart(outImg.image, auxParam, "wrk");
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
    }
};

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
class RenderClearTimestampsFunc : public StatefulCancellableTaskFuncInterface
{
    void call(
        boost::any& threadLocalData,
        const pany_range& /*out*/,
        const const_pany_range& /*in*/,
        const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        const override
    {
        auto& localState =
            boost::any_cast<RenderThreadLocalState&>(threadLocalData);
        localState.vsWorker->clearTimestamps();
    }
};

class RenderEnableNvProfFunc : public StatefulCancellableTaskFuncInterface
{
    void call(
        boost::any& threadLocalData,
        const pany_range& /*out*/,
        const const_pany_range& /*in*/,
        const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        const override
    {
        auto& localState =
            boost::any_cast<RenderThreadLocalState&>(threadLocalData);
	if (localState.threadId == 0)
    	    localState.vsWorker->enableNvProf();
    }
};
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

class AssembleFunc : public StatefulCancellableTaskFuncInterface
{
public:
    void call(
        boost::any& threadLocalData,
        const pany_range& out,
        const const_pany_range& in,
        const silver_bullets::sync::CancelController::Checker& isCancelled)
        const override
    {
        auto& localState =
            boost::any_cast<AssembleThreadLocalState&>(threadLocalData);
        boost::ignore_unused(localState);
        S3VS_TIME_ESTIMATOR(estimator, localState.workerTimeInfo->timeData)

        // Compute bounding box for the resulting image
        BoundingBox<2, int> bb;
        // Use only two params: all the next may have another meaning (TaskAuxParam)!
        auto inParts = boost::make_iterator_range(in.begin(), in.begin() + 2);
        for (auto inputItem: inParts)
            bb << boost::any_cast<const RgbaImagePart&>(*inputItem);

        if (isCancelled)
            return;

        // Lay out resulting image
        *out[0] = RgbaImagePart();
        auto& result = boost::any_cast<RgbaImagePart&>(*out[0]);
        result.origin = bb.min();
        result.image.size = bb.size();
        auto W = result.image.size[0];
        auto H = result.image.size[1];
        result.image.bits.resize(W * H * 4, 0);

        auto& sharedState = boost::any_cast<const VsRenderSharedStateEx&>(
            *readOnlySharedData());
        auto calcMaxIntProjection =
            sharedState.input.fieldMode() == FieldMode::MaxIntensityProjection;
        if (calcMaxIntProjection)
        {
            auto p = reinterpret_cast<float*>(result.image.bits.data());
            std::fill(p, p + W*H, nanf(""));
        }

        // Blend input images to the output
        bool justCopySrc = true; // first image part is just copied
        for (auto inputItem: inParts)
        {
            if (isCancelled)
                return;
            if (calcMaxIntProjection)
                blendImagePartMaxIntensity(
                    result.image,
                    boost::any_cast<const RgbaImagePart&>(*inputItem),
                    justCopySrc,
                    result.origin);
            else
                blendImagePart(
                    result.image,
                    boost::any_cast<const RgbaImagePart&>(*inputItem),
                    justCopySrc,
                    result.origin);
            justCopySrc = false;
        }

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
        auto& auxParam = boost::any_cast<const TaskAuxParam&>(*in[2]);
        saveImagePart(result.image, auxParam, "asm");
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
    }
};

class FinalComposeFunc : public StatefulCancellableTaskFuncInterface
{
public:
    void call(
        boost::any& threadLocalData,
        const pany_range& /*out*/,
        const const_pany_range& in,
        const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        const override
    {
        auto& localState =
            boost::any_cast<AssembleThreadLocalState&>(threadLocalData);
        boost::ignore_unused(localState);
        S3VS_TIME_ESTIMATOR(estimator, localState.workerTimeInfo->timeData)

        // Compute viewport size
        auto& sharedState = boost::any_cast<const VsRenderSharedStateEx&>(
            *readOnlySharedData());
        auto viewportSize = sharedState.input.viewportSize();
        auto W = viewportSize[0];
        auto H = viewportSize[1];

        auto calcMaxIntProjection =
            sharedState.input.fieldMode() == FieldMode::MaxIntensityProjection;

        // Finally compose output frame
        auto& composedImage = *boost::any_cast<RgbaImage*>(*in[1]);
        auto pixelCount = W*H;
        composedImage.bits.resize(pixelCount << 2);
        auto pixels = reinterpret_cast<unsigned int*>(composedImage.bits.data());
        auto backgroundColor = floatRgbToUint32Argb(sharedState.input.backgroundColor());
        std::fill(pixels, pixels + pixelCount, backgroundColor);
        composedImage.size = viewportSize;

        const auto& imageSrc = boost::any_cast<const RgbaImagePart&>(*in[0]);
        if (calcMaxIntProjection)
            makeMaxIntensityProjection(composedImage, imageSrc);
        else
            blendImagePart(composedImage, imageSrc, false);

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
        auto& auxParam = boost::any_cast<const TaskAuxParam&>(*in[2]);
        saveImagePart(composedImage, auxParam, "final");
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
    }

    void invalidateColorTransferVector()
    {
        m_colorTransferVectorValid = false;
    }
private:
    mutable std::vector<s3dmm::Vec<4, float>> m_colorTransferVector;
    mutable bool m_colorTransferVectorValid{false};

    const std::vector<s3dmm::Vec<4, float>>& getColorTransferVector() const
    {
        if (!m_colorTransferVectorValid)
        {
            auto& sharedState = boost::any_cast<const VsRenderSharedStateEx&>(
                *readOnlySharedData());
            m_colorTransferVector = s3vs::makeColorTransferVectorVec4(
                sharedState.input.fieldAllParam().colorTransferFunction(), 512u);
            m_colorTransferVectorValid = true;
        }
        return m_colorTransferVector;
    }

    void makeMaxIntensityProjection(RgbaImage& dst, const RgbaImagePart& src) const
    {
        BOOST_ASSERT(imageBoundingBox(dst).contains(
            imagePartBoundingBox(src)));
        auto spix = reinterpret_cast<const float*>(src.image.bits.data());
        auto sw = src.image.size[0];
        auto sh = src.image.size[1];
        auto dpix = reinterpret_cast<uint32_t*>(dst.bits.data());
        auto dw = dst.size[0];
        auto dx = src.origin[0];
        auto dy = src.origin[1];
        for (auto sy = 0; sy < sh; ++sy)
        {
            auto spixRow = spix + sy*sw;
            auto dpixRow = dpix + dw * (dy + sy) + dx;
            for (auto i = 0; i < sw; ++i)
            {
                const auto& s = spixRow[i];
                if (std::isnan(s))
                    continue;
                auto rgba = s3vs::calcColor(getColorTransferVector(), s);
                dpixRow[i] = floatRgbaToUint32Argb(rgba);
            }
        }
    }
};


VsRenderer::VsRenderer(
    const VsControllerInput* input,
    ComputingCaps& compuCaps,
    VsControllerOpts& opts) :
  m_input(input),
  m_drawWorkersSharedState(VsRenderSharedStateEx()),
  m_executor(m_cancelController.canceller()),
  m_mainExecutor(m_cancelController.canceller()),
  m_compuCaps(compuCaps),
  m_opts(opts)
{
    compuCaps.onCompNodeCountChanged([this]() {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "VsRenderer: Unable to change computing nodes count "
                "when render workers are already created.");
    });
    compuCaps.onCPUPerNodeCountChanged([this]() {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "VsRenderer: Unable to change CPUs per node count "
                "when render workers are already created.");
    });
    compuCaps.onGPUPerNodeCountChanged([this]() {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "VsRenderer: Unable to change GPUs per node count "
                "when render workers are already created.");
    });
    compuCaps.onWorkerThreadPerNodeCountChanged([this]() {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "VsRenderer: Unable to change worker threads per node count "
                "when render workers are already created.");
    });

    opts.onMeasureRenderingTimeChanged([this]() {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "VsRenderer: Unable to change measureRenderingTime option "
                "when render workers are already created.");
    });

    m_mainTaskFuncRegistry[MainTask::UpdateRenderLocalState] =
        [this](
            boost::any& /*threadLocalData*/,
            const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
        {
            updateRenderLocalState();
        };

    m_mainTaskFuncRegistry[MainTask::ContinueRendering] =
        [this](
            boost::any& /*threadLocalData*/,
            const silver_bullets::sync::CancelController::Checker& isCancelled)
        {
            continueRendering(isCancelled);
        };

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_mainTaskFuncRegistry[MainTask::ClearRenderTimestamps] =
        [this](
            boost::any& /*threadLocalData*/,
            const silver_bullets::sync::CancelController::Checker& /*isCancelled*/)
    {
        clearRenderTimestamps();
    };
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION


    m_mainExecutor.setTaskFuncRegistry(&m_mainTaskFuncRegistry);
}

VsRenderer::~VsRenderer()
{
    m_cancelController.cancel();
}

void VsRenderer::setShaderPath(const std::string& shaderPath)
{
    if (renderSharedState().shaderPath != shaderPath)
    {
        if (m_vsWorkersInitialized)
            throw runtime_error(
                "Cannot change shader path because VS worker is already initialized");
        renderSharedState().shaderPath = shaderPath;
    }
}

void VsRenderer::start()
{
    maybeInitWorkers();
}

std::string VsRenderer::shaderPath() const
{
    return renderSharedState().shaderPath;
}

void VsRenderer::setCameraTransform(const Matrix4r& cameraTransform)
{
    updateRenderSharedState([&cameraTransform](VsRenderSharedStateEx& st) {
        st.cameraTransform = cameraTransform;
        st.renderStateChanges.inc(RenderStateChange::CameraPosition);
    });
}

Matrix4r VsRenderer::cameraTransform() const
{
    return renderSharedState().cameraTransform;
}

vector<string> VsRenderer::fieldNames() const
{
    return renderSharedState().fieldSvc->fieldNames();
}

Vec2r VsRenderer::fieldRange(const std::string& fieldName) const
{
    auto& fieldSvc = *renderSharedState().fieldSvc;
    auto fieldIndex = fieldSvc.fieldIndex(fieldName);
    return fieldSvc.fieldRange(fieldIndex);
}

unsigned int VsRenderer::timeFrameCount() const
{
    auto& fieldSvc = *renderSharedState().fieldSvc;
    return fieldSvc.timeFrameCount();
}

VsControllerFrameOutput VsRenderer::frameOutput() const
{
    SyncVsControllerFrameOutput syncFrameOutput(
        m_frameOutput, m_frameOutputMutex);
    maybeInitFrameOutput(syncFrameOutput, m_input->viewportSize());
    return m_frameOutput;
}

void VsRenderer::onProblemPathChanged()
{
    cout << "~~~~~~~~~~~~~ Setting problem path ..." << endl; // deBUG, TODO:
                                                              // Remove

    auto path = m_input->problemPath();
    updateRenderSharedState([&path](VsRenderSharedStateEx& st) {
        st.fieldSvc = make_shared<BlockTreeFieldService<3>>(path);
        st.primaryFieldIndex =
            resolveField(*st.fieldSvc, static_cast<WithPrimaryField&>(st.input));
        st.secondaryFieldIndex = resolveField(
            *st.fieldSvc,
            static_cast<WithSecondaryField&>(st.input.fieldAllParam()));
        st.renderStateChanges.inc(
            RenderStateChange::ProblemPath,
            RenderStateChange::Field,
            RenderStateChange::TimeFrame,
            RenderStateChange::FieldMode,
            RenderStateChange::CameraPosition,
            RenderStateChange::ColorTransfer,
            RenderStateChange::FieldParam,
            RenderStateChange::ClippingPlanes,
            RenderStateChange::ViewportSize,
            RenderStateChange::BackgroundColor,
            RenderStateChange::CameraFovY,
            RenderStateChange::RenderPatience,
            RenderStateChange::RenderQuality,
            RenderStateChange::RenderLevel);
    });
}

void VsRenderer::onTimeFrameChanged()
{
    auto timeFrame = m_input->timeFrame();
    updateRenderSharedState([&timeFrame](VsRenderSharedStateEx& st) {
        st.input.setTimeFrame(timeFrame);
        st.renderStateChanges.inc(RenderStateChange::TimeFrame);
    });
}

void VsRenderer::onPrimaryFieldChanged()
{
    auto primaryField = m_input->primaryField();
    updateRenderSharedState([&primaryField](VsRenderSharedStateEx& st) {
        st.input.setPrimaryField(primaryField);
        if (st.fieldSvc)
            st.primaryFieldIndex = resolveField(
                *st.fieldSvc, static_cast<WithPrimaryField&>(st.input));
        st.renderStateChanges.inc(RenderStateChange::Field);
    });
}

void VsRenderer::onFieldModeChanged()
{
    auto fieldMode = m_input->fieldMode();
    updateRenderSharedState([&fieldMode](VsRenderSharedStateEx& st) {
        st.input.setFieldMode(fieldMode);
        st.renderStateChanges.inc(RenderStateChange::FieldMode);
    });
}

void VsRenderer::onClippingPlanesChanged()
{
    auto clippingPlanes = m_input->clippingPlanes();
    updateRenderSharedState([&clippingPlanes](VsRenderSharedStateEx& st) {
        st.input.setClippingPlanes(clippingPlanes);
        st.renderStateChanges.inc(RenderStateChange::ClippingPlanes);
    });
}

void VsRenderer::onViewportSizeChanged()
{
    auto viewportSize = m_input->viewportSize();
    updateRenderSharedState([&viewportSize](VsRenderSharedStateEx& st) {
        st.input.setViewportSize(viewportSize);
        st.renderStateChanges.inc(RenderStateChange::ViewportSize);
    });
}

void VsRenderer::onBackgroundColorChanged()
{
    auto backgroundColor = m_input->backgroundColor();
    updateRenderSharedState([&backgroundColor](VsRenderSharedStateEx& st) {
        st.input.setBackgroundColor(backgroundColor);
        st.renderStateChanges.inc(RenderStateChange::BackgroundColor);
    });
}

void VsRenderer::onFovYChanged()
{
    auto fovY = m_input->fovY();
    updateRenderSharedState([&fovY](VsRenderSharedStateEx& st) {
        st.input.setFovY(fovY);
        st.renderStateChanges.inc(RenderStateChange::CameraFovY);
    });
}

void VsRenderer::onRenderPatienceChanged()
{
    auto renderPatience = m_input->renderPatience();
    updateRenderSharedState([&renderPatience](VsRenderSharedStateEx& st) {
        st.input.setRenderPatience(renderPatience);
        st.renderStateChanges.inc(RenderStateChange::RenderPatience);
    });
}

void VsRenderer::onRenderQualityChanged()
{
    auto renderQuality = m_input->renderQuality();
    updateRenderSharedState([&renderQuality](VsRenderSharedStateEx& st) {
        st.input.setRenderQuality(renderQuality);
        st.renderStateChanges.inc(RenderStateChange::RenderQuality);
    });
}

void VsRenderer::onRenderLevelChanged()
{
    auto renderLevel = m_input->renderLevel();
    updateRenderSharedState([&renderLevel](VsRenderSharedStateEx& st) {
        st.input.setRenderLevel(renderLevel);
        st.renderStateChanges.inc(RenderStateChange::RenderLevel);
    });
}

void VsRenderer::onFieldThresholdChanged()
{
    auto threshold = m_input->fieldAllParam().threshold();
    updateRenderSharedState([&threshold](VsRenderSharedStateEx& st) {
        st.input.fieldAllParam().setThreshold(threshold);
        st.renderStateChanges.inc(RenderStateChange::FieldParam);
    });
}

void VsRenderer::onFieldColorTransferFunctionChanged()
{
    auto colorTransferFunction =
        m_input->fieldAllParam().colorTransferFunction();
    updateRenderSharedState(
        [&colorTransferFunction](VsRenderSharedStateEx& st) {
            st.input.fieldAllParam().setColorTransferFunction(
                colorTransferFunction);
            st.renderStateChanges.inc(RenderStateChange::FieldParam);
        });
}

void VsRenderer::onFieldIsosurfaceLevelChanged()
{
    auto isosurfaceLevel = m_input->fieldAllParam().isosurfaceLevel();
    updateRenderSharedState([&isosurfaceLevel](VsRenderSharedStateEx& st) {
        st.input.fieldAllParam().setIsosurfaceLevel(isosurfaceLevel);
        st.renderStateChanges.inc(RenderStateChange::FieldParam);
    });
}

void VsRenderer::onFieldIsosurfaceLevelsChanged()
{
    auto isosurfaceLevels = m_input->fieldAllParam().isosurfaceLevels();
    updateRenderSharedState([&isosurfaceLevels](VsRenderSharedStateEx& st) {
        st.input.fieldAllParam().setIsosurfaceLevels(isosurfaceLevels);
        st.renderStateChanges.inc(RenderStateChange::FieldParam);
    });
}

void VsRenderer::onFieldIsosurfaceOpacityChanged()
{
    auto isosurfaceOpacity = m_input->fieldAllParam().isosurfaceOpacity();
    updateRenderSharedState([&isosurfaceOpacity](VsRenderSharedStateEx& st) {
        st.input.fieldAllParam().setIsosurfaceOpacity(isosurfaceOpacity);
        st.renderStateChanges.inc(RenderStateChange::FieldParam);
    });
}

void VsRenderer::onFieldSecondaryFieldChanged()
{
    auto secondaryField = m_input->fieldAllParam().secondaryField();
    updateRenderSharedState([&secondaryField](VsRenderSharedStateEx& st) {
        st.input.fieldAllParam().setSecondaryField(secondaryField);
        if (st.fieldSvc)
            st.secondaryFieldIndex = resolveField(
                *st.fieldSvc,
                static_cast<WithSecondaryField&>(st.input.fieldAllParam()));
        st.renderStateChanges.inc(RenderStateChange::FieldParam);
    });
}

VsRenderSharedStateEx& VsRenderer::renderSharedState()
{
    return boost::any_cast<VsRenderSharedStateEx&>(m_drawWorkersSharedState);
}

const VsRenderSharedStateEx& VsRenderer::renderSharedState() const
{
    return boost::any_cast<const VsRenderSharedStateEx&>(
        m_drawWorkersSharedState);
}

void VsRenderer::maybeInitWorkers()
{
    if (m_vsWorkersInitialized)
        return;
    m_vsWorkersInitialized = true;

    m_taskFuncRegistry[GraphTask::RenderStateInit] =
        make_shared<RenderStateInitFunc>();
    m_taskFuncRegistry[GraphTask::RenderStateUpdate] =
        make_shared<RenderStateUpdateFunc>();
    m_taskFuncRegistry[GraphTask::Render] = make_shared<RenderFunc>();
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_taskFuncRegistry[GraphTask::ClearTimestamps] =
        make_shared<RenderClearTimestampsFunc>();
    m_taskFuncRegistry[GraphTask::EnableNvProf] =
        make_shared<RenderEnableNvProfFunc>();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_taskFuncRegistry[GraphTask::Assemble] = make_shared<AssembleFunc>();
    auto finalComposeFunc = make_shared<FinalComposeFunc>();
    m_taskFuncRegistry[GraphTask::FinalCompose] = finalComposeFunc;

    auto& sharedState = renderSharedState();
    sharedState.input = *m_input;
    sharedState.input.fieldAllParam().onColorTransferFunctionChanged(
        [finalComposeFunc](){
            finalComposeFunc->invalidateColorTransferVector();
    });

    auto nodeCount = m_compuCaps.compNodeCount();
    if (nodeCount == 0)
        throw runtime_error(
            "VsRenderer::maybeInitWorkers: computing nodes count is 0");
    auto gpuPerNode = m_compuCaps.GPUPerNodeCount();
    if (gpuPerNode == 0)
        throw runtime_error(
            "VsRenderer::maybeInitWorkers: GPU count per node is 0");
    auto cpuPerNode = m_compuCaps.CPUPerNodeCount();
    if (cpuPerNode == 0)
        throw runtime_error(
            "VsRenderer::maybeInitWorkers: CPU count per node is 0");
    auto workerThreadPerNode =
        m_compuCaps.workerThreadPerNodeCount() ? m_compuCaps.workerThreadPerNodeCount() : gpuPerNode;

    m_indexCubeSplitStorage.setSplitCount(workerThreadPerNode * nodeCount);
    m_renderGraphGenerators.init(nodeCount, workerThreadPerNode);

    auto addWorkerTimeInfo = [this](size_t node, size_t threadId, size_t resourceId){
        auto w = std::make_shared<WorkerTimeInfo>();
        w->nodeId = node;
        w->threadId = threadId;
        w->resourceType = resourceId;
        m_workerTimeInfo.push_back(w);
        return w;
    };

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_vsWorkerTimestamps.resize(nodeCount * workerThreadPerNode);
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    for (size_t node = 0; node < nodeCount; ++node)
    {
        for (size_t threadId = 0; threadId < workerThreadPerNode; ++threadId)
        {
            auto workerTimeInfo = addWorkerTimeInfo(node, threadId, RenderResourceId);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
            auto& vsWorkerTimestamps = m_vsWorkerTimestamps[node*workerThreadPerNode + threadId];
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
            auto tx = std::make_shared<TTX>(
                taskResourceIdOf(node, RenderResourceId),
                &m_taskFuncRegistry,
                [threadId, gpuPerNode, workerTimeInfo
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
                 , &vsWorkerTimestamps
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
                ]() {
                    // Init render resources
                    RenderThreadLocalState state;
                    state.threadId = threadId;
                    state.gpuId = threadId % gpuPerNode;
                    state.workerTimeInfo = workerTimeInfo.get();
                    state.vsWorker = VsWorkerInterface::newInstance("Default");
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
                    vsWorkerTimestamps = &state.vsWorker->timestamps();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
                    return boost::any(state);
                },
                m_cancelController.checker());
            tx->setReadOnlySharedData(&m_drawWorkersSharedState);
            tx->start(
                Task{0, 0, GraphTask::RenderStateInit},
                pany_range(),
                const_pany_range());
            m_drawWorkers.push_back(tx);
            m_executor.addTaskExecutor(tx);
        }
    }

    for (auto& worker: m_drawWorkers)
        worker->wait();

    for (size_t node = 0; node < nodeCount; ++node)
    {
        for (size_t threadId = 0; threadId < cpuPerNode; ++threadId)
        {
            auto workerTimeInfo = addWorkerTimeInfo(node, threadId, AssembleResourceId);
            auto tx = std::make_shared<TTX>(
                taskResourceIdOf(node, AssembleResourceId),
                &m_taskFuncRegistry,
                [threadId, workerTimeInfo]() {
                    AssembleThreadLocalState state;
                    state.threadId = threadId;
                    state.workerTimeInfo = workerTimeInfo.get();
                    return boost::any(state);
                },
                m_cancelController.checker());
            tx->setReadOnlySharedData(&m_drawWorkersSharedState);
            m_assembleWorkers.push_back(tx);
            m_executor.addTaskExecutor(tx);
        }
    }
}

void VsRenderer::updateRenderLocalState() const
{
    for (auto& worker: m_drawWorkers)
        worker->start(
            Task{0, 0, GraphTask::RenderStateUpdate},
            pany_range(),
            const_pany_range());
    for (auto& worker: m_drawWorkers)
        worker->wait();
}

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
void VsRenderer::clearRenderTimestamps() const
{
    for (auto& worker: m_drawWorkers)
        worker->start(
            Task{0, 0, GraphTask::ClearTimestamps},
            pany_range(),
            const_pany_range());
    for (auto& worker: m_drawWorkers)
        worker->wait();
}

void VsRenderer::enableNvprof() const
{
    for (auto& worker: m_drawWorkers)
        worker->start(
            Task{0, 0, GraphTask::EnableNvProf},
            pany_range(),
            const_pany_range());
    for (auto& worker: m_drawWorkers)
        worker->wait();
}

#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

void VsRenderer::continueRendering(const silver_bullets::sync::CancelController::Checker &isCancelled)
{
    const auto& sharedState = renderSharedState();
    if (!sharedState.fieldSvc)
        return;
    auto& md = sharedState.fieldSvc->metadata();
    auto levelCount = md.maxFullLevel() + 1;

    auto dir = [&]() {
        vl::Vector3<s3dmm::real_type> eye, at, up, right;
        sharedState.cameraTransform.getAsLookAt(eye, at, up, right);
        return s3dmm::Vec3d({eye[0] - at[0], eye[1] - at[1], eye[2] - at[2]});
    }();

    auto minLevel = 0u;
    if (m_input->renderLevel() >= 0) {
        minLevel =  min(static_cast<unsigned int>(m_input->renderLevel()), levelCount-1);
        levelCount = minLevel + 1;
    }

    m_indexCubeSplitStorage.setDirection(dir);

    for (size_t level = minLevel; (level < levelCount) && !isCancelled; ++level)
    {
        auto curTime = std::chrono::steady_clock::now();
        auto boxes = m_indexCubeSplitStorage.getBoxesSorted(level);
        if (boxes.empty())
            continue;
        auto& graphGen =
            m_renderGraphGenerators.getGenerator(boxes.size());
        auto graph = graphGen.getRenderGraph();
        auto workers = graphGen.getWorkerTasks();
        BOOST_ASSERT(workers.size() == boxes.size());
        for (size_t i = 0; i < workers.size(); ++i)
        {
            graph.input(workers[i], 0) = RenderFunc::Input();
            auto& renderInput = boost::any_cast<RenderFunc::Input&>(
                graph.input(workers[i], 0));
            renderInput.level = level;
            renderInput.indexBox << boxes[i].min();
            auto v = boxes[i].max();
            // VsWorkerInterface includes max into the range, when
            // m_indexCubeSplitStorage does not
            v -= 1;
            renderInput.indexBox << v;
#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
            graph.input(workers[i], 1) = TaskAuxParam{
                m_frameNumberSaving,
                static_cast<unsigned int>(i),
                static_cast<unsigned int>(level)
            };
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
        }

#ifdef S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING
        {
            unsigned int taskIndex = 0;
            for (size_t i = 0; i < graph.taskInfo.size(); ++i)
            {
                if (i == graphGen.getComposeTask())
                    continue;
                if (resourceIdOf(graph.taskInfo[i].task.resourceType) != AssembleResourceId)
                    continue;
                graph.input(i, 2) = TaskAuxParam{
                    m_frameNumberSaving,
                    taskIndex++,
                    static_cast<unsigned int>(level)
                };
            }
        }
        graph.input(graphGen.getComposeTask(), 2) = TaskAuxParam{
            m_frameNumberSaving,
            0,
            static_cast<unsigned int>(level)
        };
        ++m_frameNumberSaving;
#endif//S3DMM_ENABLE_FRAME_IMAGE_PARTS_SAVING

        graph.input(graphGen.getComposeTask(), 1) = &m_composedImage;

        auto cache = m_executor.makeCache();
        m_executor.start(&graph, cache);
        m_executor.wait();

        uint64_t time = 0;
        if (!isCancelled)
        {
            if (m_opts.measureRenderingTime())
            {
                std::chrono::duration<double, std::milli> elapsed =
                    std::chrono::steady_clock::now() - curTime;
                time = static_cast<uint64_t>(lround(elapsed.count()));
            }
            VsControllerFrameOutputRW writer(frameOutput());
            writer.writeFrame({
                                  m_frameNumber++,
                                  {
                                      time,
                                      static_cast<unsigned int>(level)
                                  }},
                              m_composedImage.bits.size() ? m_composedImage.bits.data() : nullptr);
        }
        //cout << "level " << level << (isCancelled ? " cancelled" : " processed")
        //     << " (of " << levelCount << ")" << endl;
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        writeWorkerTimeData(time, curTime, level, isCancelled);
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    }
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    if (!isCancelled)
        writeRenderWorkerTimestamps();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    if (!isCancelled && !m_firstFrameRendered) {
        m_firstFrameRendered = true;
        enableNvprof();
    }
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
}

template <class Upd>
void VsRenderer::updateRenderSharedState(const Upd& upd)
{
    m_cancelController.cancel();
    m_mainExecutor.wait();
    upd(renderSharedState());
    m_cancelController.resume();
    m_mainExecutor.post(MainTask::UpdateRenderLocalState);
    m_mainExecutor.wait();
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_mainExecutor.post(MainTask::ClearRenderTimestamps);
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_mainExecutor.post(MainTask::ContinueRendering);
}

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION

void VsRenderer::writeWorkerTimeData(
    uint64_t time,
    const std::chrono::steady_clock::time_point& curTime,
    unsigned int level,
    bool isCancelled) const
{
    for (auto& worker : m_workerTimeInfo)
        worker->callCountTotal += worker->timeData.callCount;
    if (!isCancelled)
    {
        ofstream f("VsRendererWorkerTime.log", ios_base::out|ios_base::app);
        if (f.is_open())
        {
            for (auto& worker : m_workerTimeInfo)
            {
                worker->callCountTotalSuccess += worker->timeData.callCount;
                f << "timestamp=" << curTime.time_since_epoch().count()
                  << " level=" << level
                  << " node=" << worker->nodeId
                  << " thread=" << worker->threadId
                  << " resourceId=" << worker->resourceType
                  << " time=" << worker->timeData.time
                  << " callCount=" << worker->timeData.callCount
                  << " callCountTotal=" << worker->callCountTotal
                  << " callCountTotalSuccess=" << worker->callCountTotalSuccess
                  << " timeFull=" << time;
                f << endl;
            }

        }
        else
            cout << "Failed to open VsRendererWorkerTime.log" << endl;
    }
    for (auto& worker : m_workerTimeInfo)
    {
        worker->timeData = TimeEstimatorData();
    }
}

void VsRenderer::writeRenderWorkerTimestamps() const
{
    ofstream f("VsRendererDrawTimestamps.log", ios_base::out|ios_base::app);
    if (f.is_open())
    {
        auto nodeCount = m_compuCaps.compNodeCount();
        BOOST_ASSERT(nodeCount > 0);
        auto gpuPerNode = m_compuCaps.GPUPerNodeCount();
        auto workerThreadPerNode =
            m_compuCaps.workerThreadPerNodeCount() ? m_compuCaps.workerThreadPerNodeCount() : gpuPerNode;
        for (size_t node = 0; node < nodeCount; ++node)
        {
            for (size_t threadId = 0; threadId < workerThreadPerNode; ++threadId)
            {
                auto vsWorkerTimestamps = m_vsWorkerTimestamps[node*workerThreadPerNode + threadId];
                for (auto& wts : *vsWorkerTimestamps) {
                    f << node << '\t' << threadId << '\t' << wts << endl;
                }
            }
        }
        f << "-" << endl;
    }
}

#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

} // namespace s3vs
