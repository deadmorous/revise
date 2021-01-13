#include "s3dmm_cuda/Device3DArray.hpp"
#include "s3dmm_cuda/DeviceArray.hpp"

#include "DeviceTextureObject.hpp"
#include "cudaCheck.hpp"

#include <boost/assert.hpp>

namespace s3dmm {

__host__ DeviceTextureObject::~DeviceTextureObject() {
    free();
}

__host__ void DeviceTextureObject::createBoundTexture(
    const Device3DArray& array,
    bool normalizedCoords,
    cudaTextureFilterMode filterMode,
    cudaTextureAddressMode addressMode)
{
    free();

    // https://stackoverflow.com/questions/24981310/cuda-create-3d-texture-and-cudaarray3d-from-device-memory
    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array  = array.handle();
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.addressMode[1] = addressMode;
    texDescr.addressMode[2] = addressMode;
    texDescr.readMode = cudaReadModeElementType;
    CU_CHECK(cudaCreateTextureObject(&m_t, &texRes, &texDescr, nullptr));
    BOOST_ASSERT(m_t);
}

/*
__host__ void DeviceTextureObject::createBoundTexture(
    const DeviceVector<unsigned char>& v,
    bool normalizedCoords,
    cudaTextureFilterMode filterMode,
    cudaTextureAddressMode addressMode)
{
    free();

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeLinear;
    texRes.res.linear.devPtr = const_cast<unsigned char*>(v.data());
    texRes.res.linear.desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    texRes.res.linear.sizeInBytes = v.size() * sizeof(char);
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.readMode = cudaReadModeElementType;
    CU_CHECK(cudaCreateTextureObject(&m_t, &texRes, &texDescr, nullptr));
    BOOST_ASSERT(m_t);
}

__host__ void DeviceTextureObject::createBoundTexture(
    const DeviceVector<float>& v,
    bool normalizedCoords,
    cudaTextureFilterMode filterMode,
    cudaTextureAddressMode addressMode)
{
    free();

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeLinear;
    texRes.res.linear.devPtr = const_cast<float*>(v.data());
    texRes.res.linear.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    texRes.res.linear.sizeInBytes = v.size() * sizeof(float);
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.readMode = cudaReadModeElementType;
    // texDescr.readMode = cudaReadModeNormalizedFloat;
    CU_CHECK(cudaCreateTextureObject(&m_t, &texRes, &texDescr, nullptr));
    BOOST_ASSERT(m_t);
}
*/

__host__ void DeviceTextureObject::createBoundTexture(
    const DeviceArray& array,
    bool normalizedCoords,
    cudaTextureFilterMode filterMode,
    cudaTextureAddressMode addressMode)
{
    free();

    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = array.handle();
    cudaTextureDesc     texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = normalizedCoords;
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.readMode = cudaReadModeElementType;
    // texDescr.readMode = cudaReadModeNormalizedFloat;
    CU_CHECK(cudaCreateTextureObject(&m_t, &texRes, &texDescr, nullptr));
    BOOST_ASSERT(m_t);
}

__host__ void DeviceTextureObject::free()
{
    if (m_t) {
        CU_CHECK(cudaDestroyTextureObject(m_t));
        m_t = 0;
    }
}

} // s3dmm
