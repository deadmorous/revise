#include "s3dmm_cuda/DenseFieldRenderer_cu.hpp"
#include "s3dmm_cuda/DenseFieldInterpolatorArr_cu_3.hpp"
#include "s3dmm_cuda/DeviceArray.hpp"
#include "DeviceTextureObject.hpp"
#include "s3vs/ColorTransferFunction.hpp"

#include "cudaCheck.hpp"

#include "chooseKernelDim1d.hpp"
#include "RenderKernelParam.hpp"
#include "cuda_vec_op.hpp"
#include "clamp.hpp"

#include "renderIsosurface.hpp"
#include "renderIsosurfaces.hpp"
#include "renderMaxIntensityProjection.hpp"
#include "renderValueOnIsosurface.hpp"
#include "renderValueOnIsosurfaces.hpp"
#include "renderArgb.hpp"
#include "renderArgbLight.hpp"
#include "renderDomainVoxels.hpp"

#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#define NORMAL_RENDER               0
#define DEBUG_RENDER_CUBE_FACES     1

#define DEBUG_RENDER_CUBE_FACE_OPACITY 0xcc

#define RENDER_TYPE NORMAL_RENDER
// #define RENDER_TYPE DEBUG_RENDER_CUBE_FACES

#if RENDER_TYPE == DEBUG_RENDER_CUBE_FACES
#define RENDER_NEED_FIRST_FACE_NUMBER
#endif

namespace s3dmm {

namespace {

__device__ inline float intersectCubeFace(float eyeCoord, float rayCoord, float faceCoord) {
    return (faceCoord - eyeCoord) / rayCoord;
}

__device__ inline float3 rayPoint(const float3& eyePos, const float3& ray, float tau) {
    return eyePos + ray*tau;
}

__device__ inline bool checkFaceX(const float3& pos, const RenderKernelCubeParam& cube) {
    return fabs(pos.y - cube.center.y) <= cube.halfSize && fabs(pos.z - cube.center.z) <= cube.halfSize;
}

__device__ inline bool checkFaceY(const float3& pos, const RenderKernelCubeParam& cube) {
    return fabs(pos.x - cube.center.x) <= cube.halfSize && fabs(pos.z - cube.center.z) <= cube.halfSize;
}

__device__ inline bool checkFaceZ(const float3& pos, const RenderKernelCubeParam& cube) {
    return fabs(pos.x - cube.center.x) <= cube.halfSize && fabs(pos.y - cube.center.y) <= cube.halfSize;
}

template<s3vs::FieldMode fieldMode>
__global__ void renderKernel(RenderKernelParam p)
{
    RenderFuncInput in;
    in.vx = blockIdx.x * blockDim.x + threadIdx.x;
    in.vy = blockIdx.y * blockDim.y + threadIdx.y;
    if (in.vx < p.view.W && in.vy < p.view.H) {
        in.vx += p.view.x;
        in.vy += p.view.y;

        // Compute ray direction
        in.x = (-0.5f + (in.vx + 0.5f)/p.viewport.W) * p.eye.w;
        in.y = (-0.5f + (in.vy + 0.5f)/p.viewport.H) * p.eye.h;
        in.ray = p.eye.n + p.eye.e1 * in.x + p.eye.e2 * in.y;

        // Compute intersections of the ray with cube edges;
        // compute enter and exit values of parameter tau that identifies a point on the ray.
        // Note that tau=1 corresponds to a point on the screen, tau=0 - the eye, and tau>1
        // to any point we may want to render
        in.tauEnter = 1.f;
        in.tauExit = 0.f;

        constexpr float tol = 1e-10f;

#ifdef RENDER_NEED_FIRST_FACE_NUMBER
        int firstFace = -1;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
        if (fabs(in.ray.x) > tol) {
            {
                auto tau = intersectCubeFace(p.eye.pos.x, in.ray.x, p.cube.center.x - p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceX(pos, p.cube)) {
                    (in.ray.x > 0? in.tauEnter: in.tauExit) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.x > 0)
                        firstFace = 0;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
            {
                auto tau = intersectCubeFace(p.eye.pos.x, in.ray.x, p.cube.center.x + p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceX(pos, p.cube)) {
                    (in.ray.x > 0? in.tauExit: in.tauEnter) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.x <= 0)
                        firstFace = 1;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
        }

        if (fabs(in.ray.y) > tol) {
            {
                auto tau = intersectCubeFace(p.eye.pos.y, in.ray.y, p.cube.center.y - p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceY(pos, p.cube)) {
                    (in.ray.y > 0? in.tauEnter: in.tauExit) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.y > 0)
                        firstFace = 2;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
            {
                auto tau = intersectCubeFace(p.eye.pos.y, in.ray.y, p.cube.center.y + p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceY(pos, p.cube)) {
                    (in.ray.y > 0? in.tauExit: in.tauEnter) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.y <= 0)
                        firstFace = 3;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
        }

        if (fabs(in.ray.z) > tol) {
            {
                auto tau = intersectCubeFace(p.eye.pos.z, in.ray.z, p.cube.center.z - p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceZ(pos, p.cube)) {
                    (in.ray.z > 0? in.tauEnter: in.tauExit) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.z > 0)
                        firstFace = 4;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
            {
                auto tau = intersectCubeFace(p.eye.pos.z, in.ray.z, p.cube.center.z + p.cube.halfSize);
                auto pos = rayPoint(p.eye.pos, in.ray, tau);
                if (checkFaceZ(pos, p.cube)) {
                    (in.ray.z > 0? in.tauExit: in.tauEnter) = tau;
#ifdef RENDER_NEED_FIRST_FACE_NUMBER
                    if (in.ray.z <= 0)
                        firstFace = 5;
#endif // RENDER_NEED_FIRST_FACE_NUMBER
                }
            }
        }

        if (in.tauEnter < 1)
            in.tauEnter = 1;
        if (in.tauExit > in.tauEnter) {
            in.pixelIndex = (p.viewport.H - 1 - in.vy) * p.viewport.W + in.vx;
            in.n = (1 << p.primaryField.texDepth) + 1;
            in.texelSize = 2*p.cube.halfSize / in.n;
            // in.sampleCount = static_cast<int>(p.presentation.quality*in.n);
            in.sampleCount = static_cast<int>(
                p.presentation.quality*
                (in.tauExit - in.tauEnter) * dot(in.ray, in.ray) /
                (in.texelSize*(fabs(in.ray.x) + fabs(in.ray.y) + fabs(in.ray.z))));
            in.step = in.ray * ((in.tauExit - in.tauEnter) / in.sampleCount);
            in.pos = rayPoint(p.eye.pos, in.ray, in.tauEnter);
            auto texHalfSize = p.cube.halfSize + 0.5*in.texelSize;
            in.cubeOrigin = p.cube.center - float3{static_cast<float>(texHalfSize),
                    static_cast<float>(texHalfSize),
                    static_cast<float>(texHalfSize)};
            in.texCoordFactor = in.n / (2*texHalfSize);

#if RENDER_TYPE == NORMAL_RENDER
            p.viewport.pixels[in.pixelIndex] = RenderFuncCaller<fieldMode>::call(in, p);
#elif RENDER_TYPE == DEBUG_RENDER_CUBE_FACES
            if (firstFace >= 0) {
                uint32_t palette[] = {
                    0xff0000, // Red
                    0x00ff00, // Green
                    0x0000ff, // Blue
                    0x00ffff, // Cyan
                    0xff00ff, // Magenta
                    0xffff00  // Yellow
                };
                // p.viewport.pixels[in.pixelIndex] = palette[firstFace];
                auto& pix = p.viewport.pixels[in.pixelIndex];
                auto rgba_f = uint32ArgbToFloatRgba(palette[firstFace] | (DEBUG_RENDER_CUBE_FACE_OPACITY << 24));
                auto pix_f = uint32ArgbToFloatRgba(pix);
                blendAbove(pix_f, rgba_f);
                pix = floatRgbaToUint32Argb(pix_f);

            }
#endif // RENDER_TYPE
        }

        // deBUG
        // Draw red border around the view.
//        if (in.vx == p.view.x || in.vx == p.view.x + p.view.W-1 ||
//            in.vy == p.view.y || in.vy == p.view.y + p.view.H-1)
//        {
//            auto pixelIndex = (p.viewport.H - 1 - in.vy) * p.viewport.W + in.vx;
//            p.viewport.pixels[pixelIndex] = 0xffff0000;
//        }
    }
}

} // anonymous namespace


class DenseFieldRenderer_cu::Impl
{
public:
    ~Impl() {
        setRenderSharedState(nullptr);
    }

    void setRenderSharedState(const s3vs::VsRenderSharedState* sharedState)
    {
        if (m_renderSharedState != sharedState) {
            if (m_renderSharedState) {
                auto& input = const_cast<s3vs::VsControllerInput&>(m_renderSharedState->input);
                input.offPrimaryFieldChanged(m_connPrimaryField);
                input.fieldAllParam().offSecondaryFieldChanged(m_connSecondaryField);
                input.offTimeFrameChanged(m_connTimeFrame);
                input.fieldAllParam().offColorTransferFunctionChanged(m_connColorTransfer);
            }
            m_renderSharedState = sharedState;
            if (m_renderSharedState) {
                auto setPrimaryFieldDirty = [this] {
                    m_primaryField.makeDirty();
                };
                auto setSecondaryFieldDirty = [this] {
                    m_secondaryField.makeDirty();
                };
                auto setFieldsDirty = [this] {
                    m_primaryField.makeDirty();
                    m_secondaryField.makeDirty();
                };
                auto setColorTransferDirty = [this] {
                    m_colorTransferDirty = true;
                };
                auto& input = const_cast<s3vs::VsControllerInput&>(m_renderSharedState->input);
                m_connPrimaryField = input.onPrimaryFieldChanged(setPrimaryFieldDirty);
                m_connSecondaryField = input.fieldAllParam().onSecondaryFieldChanged(setSecondaryFieldDirty);
                m_connTimeFrame = input.onTimeFrameChanged(setFieldsDirty);
                m_connColorTransfer = input.fieldAllParam().onColorTransferFunctionChanged(setColorTransferDirty);
            }
            m_primaryField.makeDirty();
            m_secondaryField.makeDirty();
            m_colorTransferDirty = true;
        }
    }

    const s3vs::VsRenderSharedState* renderSharedState() const {
        return m_renderSharedState;
    }

    void clearViewport(DeviceVector<std::uint32_t>& viewport) const
    {
        auto viewportSize = m_renderSharedState->input.viewportSize();
        viewport.resize(viewportSize[0]*viewportSize[1]);
        auto fieldMode = m_renderSharedState->input.fieldMode();
        std::uint32_t val = 0u;
        if (fieldMode == s3vs::FieldMode::MaxIntensityProjection)
            *reinterpret_cast<float*>(&val) = nanf("");
        thrust::fill(thrust::device, viewport.begin(), viewport.end(), val);
    }

    void renderDenseField(
        DeviceVector<std::uint32_t>& viewport,
        unsigned int level,
        const MultiIndex<3, unsigned int>& index,
        const std::atomic<bool>& isCancelled)
    {
        m_timestamps = s3vs::BlockTimestamps();
        m_timestamps.blockIndex = index;

        // Determine field mode and if we have a secondary field
        auto fieldMode = m_renderSharedState->input.fieldMode();
        auto hasSecondaryField =
            fieldMode == s3vs::FieldMode::ValueOnIsosurface ||
            fieldMode == s3vs::FieldMode::ValueOnIsosurfaces;

        // Prepare dense field and the corresponding texture
        m_primaryField.makeField(
            m_renderSharedState,
            m_renderSharedState->primaryFieldIndex,
            level,
            index,
            m_timestamps.afterPrimaryField);
        if (hasSecondaryField)
            m_secondaryField.makeField(
                m_renderSharedState,
                m_renderSharedState->secondaryFieldIndex,
                level,
                index,
                m_timestamps.afterSecondaryField);

        // Prepare color transfer function texture
        makeColorTransferTexture();

        if (isCancelled)
            return;

        // Compute kernel parameters

        RenderKernelParam p;

        // Viewport
        auto viewportSize = m_renderSharedState->input.viewportSize();
        p.viewport.pixels = viewport.data();
        p.viewport.W = viewportSize[0];
        p.viewport.H = viewportSize[1];

        // Field
        auto fs = m_renderSharedState->fieldSvc.get();
        auto setFieldParam = [&](RenderKernelFieldParam& dst, const DeviceField& src) {
            dst.tex = src.fieldTexObj().handle();
            dst.texDepth = fs->metadata().subtreeDepth(level, index);
            dst.minValue = src.fieldRange()[0];
            dst.maxValue = src.fieldRange()[1];
        };
        setFieldParam(p.primaryField, m_primaryField);
        if (hasSecondaryField)
            setFieldParam(p.secondaryField, m_secondaryField);

        // Cube size and position
        auto cubesPerLevel = 1 << level;
        constexpr auto TopLevelCubeHalfSize = 5.f;
        auto cubeHalfSize = TopLevelCubeHalfSize / cubesPerLevel;
        auto cubeCenterCoord = [&](unsigned int idx) {
            return -TopLevelCubeHalfSize + cubeHalfSize*(1 + (idx << 1));
        };
        p.cube.halfSize = cubeHalfSize;
        p.cube.center = {
            cubeCenterCoord(index[0]),
            cubeCenterCoord(index[1]),
            cubeCenterCoord(index[2])
        };
        p.cube.level = level;

        // Camera
        constexpr auto Lcam = 0.01f * 2 * TopLevelCubeHalfSize;
        auto& tcam = m_renderSharedState->cameraTransform;
        auto tcami = tcam.getInverse();
        auto toFloat3 = [](const vl::Vector3<real_type>& x) -> float3 {
            return {x[0], x[1], x[2]};
        };
        auto normalized = [](const vl::Vector3<real_type>& x)
        {
            BOOST_ASSERT(x.length() > 0);
            auto result = x;
            result.normalize();
            return result;
        };
        p.eye.pos = toFloat3(-tcami.get3x3()*tcam.getT());
        p.eye.e1 = toFloat3(normalized(tcami.getX()));
        p.eye.e2 = toFloat3(normalized(tcami.getY()));
        p.eye.n = toFloat3(normalized(tcami.getZ()) * (-Lcam));
        p.eye.h = 2*Lcam*tan(0.5*m_renderSharedState->input.fovY()*M_PI/180);
        p.eye.pixelSize = p.eye.h / viewportSize[1];
        p.eye.w = p.eye.pixelSize * viewportSize[0];

        // View rectangle
        {
            BoundingBox<2, int> bb;
            Vec3i i;
            do {
                auto d = ((i << 1) - MultiIndex<3, int>{1,1,1})
                             .convertTo<float>()*p.cube.halfSize;
                vl::Vector4<float> vertexPosWorld(
                    p.cube.center.x + d[0],
                    p.cube.center.y + d[1],
                    p.cube.center.z + d[2],
                    1.f);
                auto vertexPosEye = tcam * vertexPosWorld;
                if (vertexPosEye.z() >= 0)
                {
                    // TODO better
                    bb << Vec2i{0, 0} << viewportSize;
                    break;
                }
                Vec2i vertexPosScreen;
                vertexPosScreen[0] =
                    clamp(
                        static_cast<int>(-vertexPosEye[0] / vertexPosEye[2] * Lcam / p.eye.pixelSize) + (p.viewport.W >> 1),
                        0, p.viewport.W - 1);
                vertexPosScreen[1] =
                    clamp(
                        static_cast<int>(-vertexPosEye[1] / vertexPosEye[2] * Lcam / p.eye.pixelSize) + (p.viewport.H >> 1),
                        0, p.viewport.H - 1);
                bb << vertexPosScreen;
            }
            while (inc01MultiIndex(i));
            p.view.x = bb.min()[0];
            p.view.y = bb.min()[1];
            p.view.W = bb.size()[0];
            p.view.H = bb.size()[1];
        }

        if (p.view.H == 0 || p.view.W == 0) {
            // Empty view rectangle
            m_timestamps.afterRender = m_timestamps.beforeRender = s3dmm::hires_time();
            return;
        }

        // Presentation parameters
        auto& fp = m_renderSharedState->input.fieldAllParam();
        switch (fieldMode) {
            case s3vs::FieldMode::Isosurface:
            case s3vs::FieldMode::ValueOnIsosurface:
                p.presentation.threshold = relToAbsFieldValue(fp.isosurfaceLevel());
                break;
            case s3vs::FieldMode::Isosurfaces:
            case s3vs::FieldMode::ValueOnIsosurfaces:
                p.presentation.threshold = 0;
                makeLevels(p.presentation, fp.isosurfaceLevels());
                break;
            case s3vs::FieldMode::Argb:
            case s3vs::FieldMode::ArgbLight:
            case s3vs::FieldMode::DomainVoxels:
            case s3vs::FieldMode::MaxIntensityProjection:
                p.presentation.threshold = relToAbsFieldValue(fp.threshold());
                break;
        }
        p.presentation.isosurfaceOpacity = fp.isosurfaceOpacity();
        p.presentation.colorTex = m_colorTransferTexObj.handle();
        p.presentation.quality = m_renderSharedState->input.renderQuality();

        // Light sources
        p.lights.sourceCount = 1;
        p.lights.sources[0].pos = p.eye.pos + p.eye.n + p.eye.e2 * p.eye.h;
        p.lights.sources[0].ambient = { 0.0, 0.0, 0.0 };
        p.lights.sources[0].diffuse = { 1.0f, 1.0f, 1.0f };
        p.lights.sources[0].specular = { 0.1f, 0.1f, 0.1f };

        // Compute kernel grid parameters
        constexpr auto maxBlockSize1dx = 32;
        constexpr auto maxBlockSize1dy = 1;
        auto dimx = chooseKernelDim1d(p.view.W, maxBlockSize1dx);
        auto dimy = chooseKernelDim1d(p.view.H, maxBlockSize1dy);
        dim3 blockCount(dimx.first, dimy.first, 1);
        dim3 blockSize(dimx.second, dimy.second, 1);

        // Run render kernel
        m_timestamps.beforeRender = s3dmm::hires_time();
        switch (fieldMode) {
            case s3vs::FieldMode::Isosurface:
                renderKernel<s3vs::FieldMode::Isosurface><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::Isosurfaces:
                renderKernel<s3vs::FieldMode::Isosurfaces><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::MaxIntensityProjection:
                renderKernel<s3vs::FieldMode::MaxIntensityProjection><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::Argb:
                renderKernel<s3vs::FieldMode::Argb><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::ArgbLight:
                renderKernel<s3vs::FieldMode::ArgbLight><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::DomainVoxels:
                renderKernel<s3vs::FieldMode::DomainVoxels><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::ValueOnIsosurface:
                renderKernel<s3vs::FieldMode::ValueOnIsosurface><<<blockCount, blockSize>>>(p);
                break;
            case s3vs::FieldMode::ValueOnIsosurfaces:
                renderKernel<s3vs::FieldMode::ValueOnIsosurfaces><<<blockCount, blockSize>>>(p);
                break;
        }
        m_timestamps.afterRender = s3dmm::hires_time();

        CU_CHECK(cudaPeekAtLastError());
    }

    const s3vs::BlockTimestamps& timestamps() const {
        return m_timestamps;
    }

private:
    const s3vs::VsRenderSharedState* m_renderSharedState = nullptr;
    s3vs::BlockTimestamps m_timestamps;

    boost::signals2::connection m_connPrimaryField;
    boost::signals2::connection m_connSecondaryField;
    boost::signals2::connection m_connTimeFrame;
    boost::signals2::connection m_connColorTransfer;

    class DeviceField
    {
    public:
        void makeField(
            const s3vs::VsRenderSharedState* renderSharedState,
            unsigned int fieldIndex,
            unsigned int level,
            const MultiIndex<3, unsigned int>& index,
            FieldTimestamps& timestamps)
        {
            auto fs = renderSharedState->fieldSvc.get();
            BOOST_ASSERT(fs);
            auto subtreeRoot = fs->metadata().blockTree().blockAt(index, level);
            auto texelsPerEdge = fs->denseFieldVerticesPerEdge(subtreeRoot);
            if (m_fieldArray.resize(texelsPerEdge, texelsPerEdge, texelsPerEdge,
                                           {32, 0, 0, 0, CudaChannelFormat::Float}))
                m_fieldTexObj.createBoundTexture(
                    m_fieldArray,
                    false,
                    cudaFilterModeLinear,
                    cudaAddressModeClamp);

            if (!m_fieldDirty && !(m_lastLevel == level && m_lastIndex == index))
                m_fieldDirty = true;
            m_lastLevel = level;
            m_lastIndex = index;

            if (m_fieldDirty) {
                // Interpolate dense field
                Vec2<dfield_real> fieldRange;
                renderSharedState->fieldSvc->interpolateWith<DenseFieldInterpolatorArr_cu_3>(
                    fieldRange,
                    m_fieldArray,
                    fieldIndex,
                    renderSharedState->input.timeFrame(),
                    subtreeRoot, &timestamps);
                m_fieldRange = renderSharedState->fieldSvc->fieldRange(fieldIndex);
                timestamps.afterGetFieldRange = s3dmm::hires_time();
                m_fieldDirty = false;
            }
            else
                timestamps.afterReadSparseField =
                    timestamps.afterComputeDenseField =
                    timestamps.afterGetFieldRange = s3dmm::hires_time();
        }

        void makeDirty() {
            m_fieldDirty = true;
        }

        const DeviceTextureObject& fieldTexObj() const {
            return m_fieldTexObj;
        }

        const Vec2<dfield_real>& fieldRange() const {
            return m_fieldRange;
        }
    private:
        Device3DArray m_fieldArray;
        Vec2<dfield_real> m_fieldRange;
        DeviceTextureObject m_fieldTexObj;
        bool m_fieldDirty = true;
        unsigned int m_lastLevel;
        MultiIndex<3, unsigned int> m_lastIndex;
    };
    DeviceField m_primaryField;
    DeviceField m_secondaryField;
    mutable std::vector<float> m_levelBuf;

    DeviceArray m_colorTransferFunction;
    DeviceTextureObject m_colorTransferTexObj;
    bool m_colorTransferDirty = true;
    void makeColorTransferTexture()
    {
        if (m_colorTransferDirty) {
            constexpr auto ColorTransferTexSize = 512u;
            auto ctf = s3vs::makeColorTransferVector(
                m_renderSharedState->input.fieldAllParam().colorTransferFunction(),
                ColorTransferTexSize);
            m_colorTransferFunction.resize(ColorTransferTexSize, 1, {32, 32, 32, 32, CudaChannelFormat::Float});
            m_colorTransferFunction.upload(ctf.data());
            m_colorTransferTexObj.createBoundTexture(
                m_colorTransferFunction,
                true,
                cudaFilterModeLinear,
                cudaAddressModeClamp);
            m_colorTransferDirty = false;
        }
    }

    dfield_real relToAbsFieldValue(dfield_real r) const
    {
        auto& range = m_primaryField.fieldRange();
        return range[0] + (range[1]-range[0]) * r;
    }

    void makeLevels(
        RenderKernelPresentationParam& dst,
        const std::vector<s3dmm::real_type>& levels) const
    {
        m_levelBuf.resize(levels.size());
        std::copy(levels.begin(), levels.end(), m_levelBuf.begin());
        std::transform(m_levelBuf.begin(), m_levelBuf.end(), m_levelBuf.begin(), [this](float level) {
            return relToAbsFieldValue(level);
        });
        std::sort(m_levelBuf.begin(), m_levelBuf.end());
        auto itEnd = std::unique(m_levelBuf.begin(), m_levelBuf.end());
        auto levelCount = static_cast<unsigned int>(itEnd - m_levelBuf.begin());
        if (levelCount > RenderKernelPresentationParam::MaxLevels)
            levelCount = RenderKernelPresentationParam::MaxLevels;
        copy(m_levelBuf.begin(), m_levelBuf.begin() + levelCount, dst.levels);
        dst.levelCount = levelCount;
    }
};

DenseFieldRenderer_cu::DenseFieldRenderer_cu() :
  m_impl(std::make_unique<Impl>())
{
}

DenseFieldRenderer_cu::~DenseFieldRenderer_cu() = default;

void DenseFieldRenderer_cu::setRenderSharedState(const s3vs::VsRenderSharedState* sharedState) {
    m_impl->setRenderSharedState(sharedState);
}

const s3vs::VsRenderSharedState* DenseFieldRenderer_cu::renderSharedState() const {
    return m_impl->renderSharedState();
}

void DenseFieldRenderer_cu::clearViewport(DeviceVector<std::uint32_t>& viewport) const {
    m_impl->clearViewport(viewport);
}

void DenseFieldRenderer_cu::renderDenseField(
    DeviceVector<std::uint32_t>& viewport,
    unsigned int level,
    const MultiIndex<3, unsigned int>& index,
    const std::atomic<bool>& isCancelled)
{
    m_impl->renderDenseField(viewport, level, index, isCancelled);
}

const s3vs::BlockTimestamps& DenseFieldRenderer_cu::timestamps() const {
    return m_impl->timestamps();
}

} // s3dmm
