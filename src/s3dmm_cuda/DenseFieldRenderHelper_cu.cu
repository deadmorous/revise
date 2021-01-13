#include "s3dmm_cuda/DenseFieldRenderHelper_cu.hpp"
#include "s3dmm_cuda/DeviceVector.hpp"
#include "s3dmm_cuda/CudaGlResourceUser.hpp"

#include "s3dmm/IndexTransform.hpp"
#include "s3dmm/BlockTreeNodeCoord.hpp"

#include "cudaCheck.hpp"
#include "glCheck.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include <boost/static_assert.hpp>

namespace s3dmm {

BOOST_STATIC_ASSERT(std::is_same<dfield_real, float>::value);

namespace {

class LinTransformNonNanToUint16 : thrust::unary_function<dfield_real, uint16_t>
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

struct ValueToAlpha : thrust::unary_function<dfield_real, uint16_t>
{
    __host__ __device__ uint16_t operator()(const dfield_real& x) const {
        return isnan(x)? 0: 0xffff;
    }
};

} // anonymous namespace

void DenseFieldRenderHelper_cu::prepareTextures(
        unsigned int depth,
        dfield_real *fieldDev,
        gpu_ll::CudaGlResource& fieldTex,
        gpu_ll::CudaGlResource& alphaTex,
        const Vec2<dfield_real>& dfieldRange)
{
    BOOST_ASSERT(fieldDev);
    auto texelsPerEdge = IndexTransform<3>::template verticesPerEdge<BlockTreeNodeCoord>(depth);
    auto fieldSize = IndexTransform<3>::vertexCount(depth);

    // Linearly transform non-NaNs in the field to ushort range [0, 0xffff]
    DeviceVector<uint16_t> fieldHalf(fieldSize);
    if (dfieldRange[1] >= dfieldRange[0]) {
        thrust::transform(thrust::device, fieldDev, fieldDev+fieldSize, fieldHalf.data(),
                          LinTransformNonNanToUint16(dfieldRange[0], dfieldRange[1]));
    }

    // Copy field to texture
    gpu_ll::CudaGlResource cudaResource;
    cudaMemcpy3DParms param = {0};
    {
        CudaGlTextureWUser<cudaArray> cudaResourceUser(fieldTex);
        param.srcPtr.ptr = fieldHalf.data();
        param.srcPtr.pitch = texelsPerEdge * sizeof(uint16_t);
        param.srcPtr.xsize = texelsPerEdge;
        param.srcPtr.ysize = texelsPerEdge;
        param.dstArray = cudaResourceUser.data();
        param.extent = { texelsPerEdge, texelsPerEdge, texelsPerEdge };
        param.kind = cudaMemcpyDeviceToDevice;
        CU_CHECK(cudaMemcpy3D(&param));
    }

    // Generate alpha, if requested
    if (alphaTex.glResource()) {
        thrust::transform(thrust::device, fieldDev, fieldDev+fieldSize, fieldHalf.data(), ValueToAlpha());
        CudaGlTextureWUser<cudaArray> cudaResourceUser(alphaTex);
        param.dstArray = cudaResourceUser.data();
        CU_CHECK(cudaMemcpy3D(&param));
    }
}

} // s3dmm
