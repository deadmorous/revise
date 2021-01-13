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

#include "VsWorker_cu.hpp"
#include "InitBlockSorter.hpp"

#include <QImage>

#include "s3dmm_cuda/selectGpu.hpp"
#include "s3dmm_cuda/manage_profiling.hpp"

#include "default_dense_field_gen.hpp"

#include "silver_bullets/fs_ns_workaround.hpp"

#include "foreach_byindex32.hpp"

#include <boost/core/ignore_unused.hpp>

namespace s3vs
{

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VsWorker_cu, "CUDA")

#ifdef S3VS_WORKER_CUDA_RENDERING
static VsWorker_cu::Registrator DefaultVsWorkerRegistrator( "Default" );
#endif // !S3VS_WORKER_CUDA_RENDERING

namespace
{
auto constexpr CHANNEL_COUNT = 4;

QRect getBoundRect(QImage qImage)
{
    QRect ofTheKing;

    int maxX = 0;
    int minX = qImage.width();
    int maxY = 0;
    int minY = qImage.height();

    auto bits = qImage.bits() + 3;
    auto h = qImage.height();
    auto w = qImage.width();

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++, bits += 4)
        {
            if (*bits != 0)
            {
                if (x < minX)
                    minX = x;
                if (x > maxX)
                    maxX = x;
                if (y < minY)
                    minY = y;
                if (y > maxY)
                    maxY = y;
            }
        }
    }

    if (minX > maxX || minY > maxY)
        return QRect(); // there are no objects in the image
    else
        ofTheKing.setCoords(minX, minY, maxX, maxY);

    return ofTheKing;
}

} // namespace

VsWorker_cu::~VsWorker_cu()
{
}

void VsWorker_cu::initialize(
        const std::string& /*shaderPath*/,
        bool /*logInfo*/,
        const std::string& /*logFileName*/,
        size_t threadId)
{
    s3dmm::selectGpu(threadId);
    boost::ignore_unused(threadId);
}

void VsWorker_cu::setRenderSharedState(const VsRenderSharedState *sharedState)
{
    m_sharedState = sharedState;
    m_fieldRenderer.setRenderSharedState(m_sharedState);
}

const VsRenderSharedState* VsWorker_cu::renderSharedState() const {
    return m_sharedState;
}

template<class To, class From>
inline vl::Matrix4<To> convert(
        const vl::Matrix4<From>& x,
        std::enable_if_t<std::is_same_v<To, From>, int> = 0)
{
    return x;
}

template<class To, class From>
inline vl::Matrix4<To> convert(
        const vl::Matrix4<From>& x,
        std::enable_if_t<!std::is_same_v<To, From>, int> = 0)
{
    vl::Matrix4<To> result;
    std::copy(x.ptr(), x.ptr()+16, result.ptr());
    return result;
}

void VsWorker_cu::updateState(unsigned int flags)
{
    if (flags & UpdateProblemPath)
        updateProblemPath();
    if (flags & UpdateViewportSize)
        updateViewportSize();
    if (flags & UpdateBackgroundColor)
        updateBackgroundColor();
    if (flags & UpdateField)
        updateField();
    if (flags & UpdateTimeFrame)
        updateTimeFrame();
    if (flags & UpdateFieldMode)
        updateFieldMode();
    if (flags & UpdateCameraPosition)
        updateCameraPosition();
    if (flags & UpdateColorTransfer)
        updateColorTransfer();
    if (flags & UpdateFieldParam)
        updateFieldParam();
    if (flags & UpdateClippingPlanes)
        updateClippingPlanes();
    if (flags & UpdateCameraFovY)
        updateCameraFovY();
    if (flags & UpdateRenderPatience)
        updateRenderPatience();
    if (flags & UpdateRenderQuality)
        updateRenderQuality();
    if (flags & UpdateRenderLevel)
        updateRenderLevel();
}

void VsWorker_cu::updateProblemPath() {
    // m_applet->volumeTextureData()->setFieldService(m_sharedState->fieldSvc.get());
}

void VsWorker_cu::updateViewportSize()
{
}

void VsWorker_cu::updateBackgroundColor()
{
}

void VsWorker_cu::updateField()
{
}

void VsWorker_cu::updateTimeFrame()
{
}

void VsWorker_cu::updateFieldMode()
{
//    auto fieldMode = m_sharedState->input.fieldMode();
}

void VsWorker_cu::updateCameraPosition()
{
//    auto camera = m_applet->rendering()->as<vl::Rendering>()->camera();
//    camera->setViewMatrix(convert<float>(m_sharedState->cameraTransform));
}

void VsWorker_cu::updateColorTransfer()
{
}

void VsWorker_cu::updateFieldParam()
{
}

void VsWorker_cu::updateClippingPlanes()
{
}

void VsWorker_cu::updateCameraFovY()
{
}

void VsWorker_cu::updateRenderPatience()
{
}

void VsWorker_cu::updateRenderQuality()
{
}

void VsWorker_cu::updateRenderLevel()
{
}


RgbaImagePart VsWorker_cu::renderScenePart(
        const SubtreeSetDescription& subtrees,
        const silver_bullets::sync::CancelController::Checker& isCancelled)
{
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    m_timestamps.emplace_back();
    auto& timestamps = m_timestamps.back();
    timestamps.level = subtrees.level;
    timestamps.start = s3dmm::hires_time();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

    RgbaImagePart imagePart;

    if (subtrees.indexBox.empty())
        return imagePart;

    // TODO: Redesign to get rid of this wild stuff
    struct UglyCancelCheckUnprotector : silver_bullets::sync::CancelController::Checker {
        using silver_bullets::sync::CancelController::Checker::cancelled;
    };
    const auto& cancelled = const_cast<UglyCancelCheckUnprotector*>(
        static_cast<const UglyCancelCheckUnprotector*>(&isCancelled))->cancelled();

    // Render
    m_fieldRenderer.clearViewport(m_viewport);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    timestamps.afterClearViewport = s3dmm::hires_time();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

    initBlockSorter(m_blockSorter, m_sharedState->cameraTransform, subtrees.indexBox);
    auto& sortedBlocks = m_blockSorter.getSortedBlocks();
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    timestamps.afterSortBlocks = s3dmm::hires_time();
    timestamps.blocks.resize(sortedBlocks.size());
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

    foreach_byindex32(iblock, sortedBlocks) {
        if (isCancelled)
            break;
        auto& index = sortedBlocks[iblock];
        m_fieldRenderer.renderDenseField(m_viewport, subtrees.level, index, cancelled);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        timestamps.blocks[iblock] = m_fieldRenderer.timestamps();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    }

    // Clip
    if (!isCancelled) {
        // Compute image part containing nonzero pixels
        auto wiewportSize = m_sharedState->input.viewportSize();
        m_viewport.download(m_viewportHostBuf);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        timestamps.afterRenderDenseField = s3dmm::hires_time();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

        QImage image(
            reinterpret_cast<unsigned char*>(m_viewportHostBuf.data()),
            wiewportSize[0],
            wiewportSize[1],
            QImage::Format_ARGB32);

        auto boundRect = getBoundRect(image);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        timestamps.afterGetClipRect = s3dmm::hires_time();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

        if (boundRect != QRect())
        {
            image = image.copy(boundRect);
            imagePart.image.bits = std::vector<unsigned char>(
                image.bits(), image.bits() + image.sizeInBytes());
            imagePart.image.size = {boundRect.width(), boundRect.height()};
            imagePart.origin = {boundRect.x(), boundRect.y()};
        }
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        timestamps.afterClip = s3dmm::hires_time();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
    }

    return imagePart;
}

#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
const std::vector<VsWorkerTimestamps>& VsWorker_cu::timestamps() const {
    return m_timestamps;
}

void VsWorker_cu::clearTimestamps() {
    m_timestamps.clear();
}

void VsWorker_cu::enableNvProf() {
    s3dmm::enableCudaProfiling();
}

#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION

} // namespace s3vs
