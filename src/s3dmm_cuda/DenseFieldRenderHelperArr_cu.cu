#include "s3dmm_cuda/DenseFieldRenderHelperArr_cu.hpp"
#include "s3dmm_cuda/DeviceVector.hpp"
#include "s3dmm_cuda/CudaGlResourceUser.hpp"

#include "s3dmm/IndexTransform.hpp"
#include "s3dmm/BlockTreeNodeCoord.hpp"

#include "cudaCheck.hpp"
#include "glCheck.hpp"
#include "surf_alg.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <boost/static_assert.hpp>

namespace s3dmm {

BOOST_STATIC_ASSERT(std::is_same<dfield_real, float>::value);

namespace {

class LinTransformNonNanToUint16
{
public:
    __host__ LinTransformNonNanToUint16(dfield_real xmin, dfield_real xmax) :
        m_xmin(xmin),
        m_factor((xmax>xmin? make_real(1/(xmax-xmin)): make_real(1))*(0x10000 - dfield_real(1e-5)))
    {}

    __host__ __device__ uint16_t operator()(const dfield_real& x) const {
        return isnan(x)? 0: static_cast<uint16_t>((x-m_xmin)*m_factor);
    }
private:
    dfield_real m_xmin;
    dfield_real m_factor;
};

struct ValueToAlpha
{
    __host__ __device__ uint16_t operator()(const dfield_real& x) const {
        return isnan(x)? 0: 0xffff;
    }
};

} // anonymous namespace

void DenseFieldRenderHelperArr_cu::prepareTextures(
        unsigned int depth,
        const Device3DArray& fieldDev,
        gpu_ll::CudaGlResource& fieldTex,
        gpu_ll::CudaGlResource& alphaTex,
        const Vec2<dfield_real>& dfieldRange)
{
    BOOST_ASSERT(fieldDev.handle());
    auto texelsPerEdge = IndexTransform<3>::template verticesPerEdge<BlockTreeNodeCoord>(depth);
    auto fieldSize = IndexTransform<3>::vertexCount(depth);

    // Linearly transform non-NaNs in the field to ushort range [0, 0xffff]
    {
        CudaGlTextureSUser<cudaArray> fieldTexUser(fieldTex);
        transform<float, uint16_t>(
                    texelsPerEdge,
                    fieldDev,
                    fieldTexUser,
                    LinTransformNonNanToUint16(dfieldRange[0], dfieldRange[1]));
    }

    // Generate alpha, if requested
    if (alphaTex.glResource()) {
        CudaGlTextureSUser<cudaArray> alphaTexUser(alphaTex);
        transform<float, uint16_t>(
                    texelsPerEdge,
                    fieldDev,
                    alphaTexUser,
                    ValueToAlpha());
    }
}

} // s3dmm
