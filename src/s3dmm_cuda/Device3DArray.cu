#include "s3dmm_cuda/Device3DArray.hpp"
#include "cudaCheck.hpp"

#include <boost/assert.hpp>
#include <cuda_runtime_api.h>

namespace s3dmm {

Device3DArray::~Device3DArray() {
    free();
}

bool Device3DArray::resize(
        std::size_t width, std::size_t height, std::size_t depth,
        const CudaChannelFormat& desc)
{
    if (!(m_a && m_extent[0] == width && m_extent[1] == height && m_extent[2] == depth && m_desc == desc)) {
        free();
        m_desc = desc;
        auto cuDesc = desc.makeCudaChannelFormatDesc();
        m_extent = { width, height, depth };
        CU_CHECK(cudaMalloc3DArray(&m_a, &cuDesc, { width, height, depth }, cudaArraySurfaceLoadStore));
        return true;
    }
    else
        return false;
}

void Device3DArray::free()
{
    if (m_surf) {
        CU_CHECK(cudaDestroySurfaceObject(m_surf));
        m_surf = 0;
    }
    if (m_a) {
        CU_CHECK(cudaFreeArray(m_a));
        m_a = nullptr;
    }
}

void Device3DArray::upload(const void *data)
{
    BOOST_ASSERT(m_a);
    cudaMemcpy3DParms param = {0};
    param.srcPos = { 0, 0, 0 };
    param.srcPtr.ptr = const_cast<void*>(data);
    auto bytesPerElement = (m_desc.x + m_desc.y + m_desc.z + m_desc.w) >> 3;
    param.srcPtr.pitch = m_extent[0] * bytesPerElement;
    param.srcPtr.xsize = m_extent[0];
    param.srcPtr.ysize = m_extent[1];
    param.dstArray = m_a;
    param.dstPos = { 0, 0, 0 };
    param.extent = { m_extent[0], m_extent[1], m_extent[2] };
    param.kind = cudaMemcpyHostToDevice;
    CU_CHECK(cudaMemcpy3D(&param));
}

void Device3DArray::download(void *data) const
{
    BOOST_ASSERT(m_a);
    cudaMemcpy3DParms param = {0};
    param.srcArray = m_a;
    param.srcPos = { 0, 0, 0 };
    param.dstPos = { 0, 0, 0 };
    param.dstPtr.ptr = data;
    param.dstPtr.pitch = m_extent[0] * m_bytesPerElement;
    param.dstPtr.xsize = m_extent[0];
    param.dstPtr.ysize = m_extent[1];
    param.extent = { m_extent[0], m_extent[1], m_extent[2] };
    param.kind = cudaMemcpyDeviceToHost;
    CU_CHECK(cudaMemcpy3D(&param));
}

cudaSurfaceObject_t Device3DArray::surface() const
{
    BOOST_ASSERT(m_a);
    if (!m_surf) {
        cudaResourceDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = m_a;
        CU_CHECK(cudaCreateSurfaceObject(&m_surf, &desc));
        BOOST_ASSERT(m_surf);
    }
    return m_surf;
}

} // s3dmm
