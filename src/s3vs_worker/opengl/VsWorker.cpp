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

#include "VsWorker.hpp"
#include "VsWorkerInit.hpp"
#include "InitBlockSorter.hpp"

#include <QImage>

#include "s3dmm/BlockTreeFieldService.hpp"
//#include "s3dmm/IncMultiIndex.hpp"

#include "default_dense_field_gen.hpp"

#include "App_MyVolumeRaycast.hpp"
#include "OpenGlSetup.hpp"
#include "gl_check.hpp"

#include <vlCore/FileSystem.hpp>
#include <vlCore/VisualizationLibrary.hpp>
#include <vlCore/GlobalSettings.hpp>

#include "silver_bullets/fs_ns_workaround.hpp"

#include <boost/core/ignore_unused.hpp>

namespace s3vs
{

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VsWorker, "OpenGL")

#ifndef S3VS_WORKER_CUDA_RENDERING
static VsWorker::Registrator DefaultVsWorkerRegistrator( "Default" );
#endif // !S3VS_WORKER_CUDA_RENDERING

namespace
{
auto constexpr CHANNEL_COUNT = 4;

void reverseImage(std::vector<unsigned char>& pixels, int width, int height)
{
    std::vector<unsigned char> line(static_cast<size_t>(CHANNEL_COUNT * width));
    auto ldata = line.data();
    unsigned char* n1 = pixels.data();
    unsigned char* n2 = pixels.data() + (height - 1) * width * CHANNEL_COUNT;
    for (; n1 < n2; n1 += CHANNEL_COUNT * width, n2 -= CHANNEL_COUNT * width)
    {
        std::copy(n1, n1 + CHANNEL_COUNT * width, ldata);
        std::copy(n2, n2 + CHANNEL_COUNT * width, n1);
        std::copy(ldata, ldata + CHANNEL_COUNT * width, n2);
    }
}

auto getPixels(int width, int height)
{
    auto size = static_cast<size_t>(CHANNEL_COUNT * width * height);
    std::vector<unsigned char> pixels(size);

    GL_CHECK(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data()));

    // Reverse the order of image lines
    reverseImage(pixels, width, height);

    return pixels;
}

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

VsWorker::~VsWorker()
{
    m_ctx->dispatchDestroyEvent();
}

void VsWorker::initialize(
        const std::string& shaderPath,
        bool logInfo,
        const std::string& logFileName,
        size_t threadId)
{
    boost::ignore_unused(threadId);

    int DefaultRenderBufferWidth = 640;
    int DefaultRenderBufferHeight = 480;
    m_oglSetup = std::make_unique<OpenGlSetup>(DefaultRenderBufferWidth, DefaultRenderBufferHeight);

    VsWorkerInit(shaderPath, logInfo, logFileName);

    {
        // m_ctx->initGLContext() changes global variables, so lock the mutex
        std::lock_guard<VsWorkerInit::mutex_type> lk(VsWorkerInit::getMutex());
        m_ctx = std::make_unique<MyOpenGlContext>();
        m_ctx->initGLContext();
    }

    auto volTexData = VolumeTextureDataInterface::newInstance(s3dmm::default_dense_field_gen);
    m_applet = new App_MyVolumeRaycast(volTexData);

    m_applet->initialize();

    m_applet->rendering()->as<vl::Rendering>()->renderer()->setFramebuffer(m_ctx->framebuffer());
    m_applet->rendering()
        ->as<vl::Rendering>()
        ->camera()
        ->viewport()
        ->setClearColor(0, 0, 0, 0);

    m_ctx->addEventListener(m_applet.get());
}

void VsWorker::setRenderSharedState(const VsRenderSharedState *sharedState) {
    m_sharedState = sharedState;
}

const VsRenderSharedState* VsWorker::renderSharedState() const {
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

void VsWorker::updateState(unsigned int flags)
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
}

void VsWorker::updateProblemPath() {
    m_applet->volumeTextureData()->setFieldService(m_sharedState->fieldSvc.get());
}

void VsWorker::updateViewportSize()
{
    auto& viewportSize = m_sharedState->input.viewportSize();
    m_oglSetup->resizeRenderBuffers(viewportSize[0], viewportSize[1]);

    auto fb = m_ctx->framebuffer();
    fb->setWidth (viewportSize[0]);
    fb->setHeight(viewportSize[1]);
    m_applet->rendering()->as<vl::Rendering>()->renderer()->setFramebuffer(fb);
    auto camera = m_applet->rendering()->as<vl::Rendering>()->camera();
    camera->viewport()->setWidth(viewportSize[0]);
    camera->viewport()->setHeight(viewportSize[1]);

    // Generate applet resize event
    m_applet->resizeEvent(viewportSize[0], viewportSize[1]);
}

void VsWorker::updateBackgroundColor()
{
}

void VsWorker::updateField()
{
}

void VsWorker::updateTimeFrame()
{
}

void VsWorker::updateFieldMode()
{
    auto fieldMode = m_sharedState->input.fieldMode();

    switch (fieldMode) {
    case FieldMode::Isosurface:
        if (m_sharedState->input.fieldParam<FieldMode::Isosurface>().isosurfaceOpacity() < s3dmm::make_real(1))
            m_applet->setRaycastMode(App_MyVolumeRaycast::Isosurface_Transp_Mode);
        else
            m_applet->setRaycastMode(App_MyVolumeRaycast::Isosurface_Mode);
        break;
    case FieldMode::Isosurfaces:
        BOOST_ASSERT(false);
        throw std::runtime_error("Field mode Isosurfaces is not implemented (TODO)");
        break;
    case FieldMode::MaxIntensityProjection:
        m_applet->setRaycastMode(App_MyVolumeRaycast::MIP_Mode);
        break;
    case FieldMode::Argb:
        m_applet->setRaycastMode(App_MyVolumeRaycast::RaycastBrightnessControl_Mode);
        break;
    case FieldMode::ArgbLight:
        m_applet->setRaycastMode(App_MyVolumeRaycast::RaycastDensityControl_Mode);
        break;
    case FieldMode::DomainVoxels:
        m_applet->setRaycastMode(App_MyVolumeRaycast::RaycastColorControl_Mode);
        break;
    case FieldMode::ValueOnIsosurface:
        BOOST_ASSERT(false);
        throw std::runtime_error("Field mode ValueOnIsosurface is not implemented (TODO)");
        break;
    case FieldMode::ValueOnIsosurfaces:
        BOOST_ASSERT(false);
        throw std::runtime_error("Field mode ValueOnIsosurfaces is not implemented (TODO)");
        break;
    }
}

void VsWorker::updateCameraPosition()
{
    auto camera = m_applet->rendering()->as<vl::Rendering>()->camera();
    camera->setViewMatrix(convert<float>(m_sharedState->cameraTransform));
}

void VsWorker::updateColorTransfer()
{
    // TODO
}

void VsWorker::updateFieldParam()
{
    // TODO: Relative threshold
    m_applet->setThresholdValue(m_sharedState->input.fieldAllParam().isosurfaceLevel());
}

void VsWorker::updateClippingPlanes()
{
    // TODO
}

void VsWorker::updateCameraFovY()
{
    // TODO
}

void VsWorker::updateRenderPatience()
{

}

void VsWorker::updateRenderQuality() {
    m_applet->setSampleCount(static_cast<unsigned int>(
        m_sharedState->input.renderQuality() * 512 + s3dmm::make_real(0.5)));
}

void VsWorker::updateRenderLevel()
{
}


RgbaImagePart VsWorker::renderScenePart(
        const SubtreeSetDescription& subtrees,
        const silver_bullets::sync::CancelController::Checker& isCancelled)
{
    RgbaImagePart imagePart;

    if (subtrees.indexBox.empty())
        return imagePart;

    m_applet->rendering()->as<vl::Rendering>()->renderer()->setClearFlags(vl::CF_CLEAR_COLOR_DEPTH);
    initBlockSorter(m_blockSorter, m_sharedState->cameraTransform, subtrees.indexBox);
    for (const auto& rootIndex: m_blockSorter.getSortedBlocks())
    {
        if (isCancelled)
            break;

        // Render image
        auto subtreeRoot = m_sharedState->fieldSvc->metadata().blockTree().blockAt(rootIndex, subtrees.level);
        m_applet->setIndex(subtreeRoot.level, subtreeRoot.location);
        m_applet->setData(
            m_sharedState->input.primaryField(),
            m_sharedState->primaryFieldIndex,
            m_sharedState->input.timeFrame(),
            subtreeRoot);

        m_ctx->update();
        m_applet->rendering()->as<vl::Rendering>()->renderer()->setClearFlags(vl::CF_DO_NOT_CLEAR);
    }

    GL_CHECK(glFinish());
    if (!isCancelled) {
        // Compute image part containing nonzero pixels
        auto wiewportSize = m_sharedState->input.viewportSize();
        auto pixels = getPixels(wiewportSize[0], wiewportSize[1]);

        QImage image(
            pixels.data(),
            wiewportSize[0],
            wiewportSize[1],
            QImage::Format_RGBA8888);

        auto boundRect = getBoundRect(image);
        if (boundRect != QRect())
        {
            image = image.copy(boundRect);
            imagePart.image.bits = std::vector<unsigned char>(
                image.bits(), image.bits() + image.sizeInBytes());
            imagePart.image.size = {boundRect.width(), boundRect.height()};
            imagePart.origin = {boundRect.x(), boundRect.y()};
        }
    }

    return imagePart;
}

} // namespace s3vs
