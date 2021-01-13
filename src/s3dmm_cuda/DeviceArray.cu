#include "s3dmm_cuda/DeviceArray.hpp"
#include "cudaCheck.hpp"

#include <boost/assert.hpp>

namespace s3dmm {

DeviceArray::~DeviceArray() {
    free();
}

bool DeviceArray::resize(
        std::size_t width, std::size_t height,
        const CudaChannelFormat& desc)
{
    if (!(m_a && m_extent[0] == width && m_extent[1] == height && m_desc == desc)) {
        free();
        m_desc = desc;
        auto cuDesc = desc.makeCudaChannelFormatDesc();
        m_extent = { width, height };
        CU_CHECK(cudaMallocArray(&m_a, &cuDesc, width, height, cudaArrayTextureGather));
        return true;
    }
    else
        return false;
}

void DeviceArray::free()
{
    if (m_a) {
        CU_CHECK(cudaFreeArray(m_a));
        m_a = nullptr;
    }
}

void DeviceArray::upload(const void *data)
{
    BOOST_ASSERT(m_a);
    auto bytesPerElement = (m_desc.x + m_desc.y + m_desc.z + m_desc.w) >> 3;
    auto spitch = m_extent[0] * bytesPerElement;
    CU_CHECK(cudaMemcpy2DToArray(
        m_a,    // cudaArray_t dst,
        0,      // size_t wOffset,
        0,      // size_t hOffset,
        data,   // const void* src,
        spitch, // size_t spitch,
        m_extent[0] * bytesPerElement,  // size_t width,
        m_extent[1],    // size_t height,
        cudaMemcpyHostToDevice  // cudaMemcpyKind kind
        ));
}

void DeviceArray::download(void *data) const
{
    BOOST_ASSERT(m_a);
    auto bytesPerElement = (m_desc.x + m_desc.y + m_desc.z + m_desc.w) >> 3;
    auto dpitch = m_extent[0] * bytesPerElement;
    CU_CHECK(cudaMemcpy2DFromArray(
        data,   // void* dst
        dpitch, // size_t dpitch
        m_a,    // cudaArray_const_t src
        0,      // size_t wOffset,
        0,      // size_t hOffset,
        m_extent[0] * bytesPerElement,  // size_t width,
        m_extent[1],    // size_t height,
        cudaMemcpyDeviceToHost  // cudaMemcpyKind kind
        ));
}

} // s3dmm
