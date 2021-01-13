#include "s3dmm_cuda/DeviceMemHandle.hpp"
#include "cudaCheck.hpp"

#include <cuda_runtime_api.h>
#include <cuda.h>

namespace s3dmm {

namespace gpu_ll {


DeviceMemHandle::DeviceMemHandle(DeviceMemHandle&& that) :
        m_d(that.m_d),
        m_byteCount(that.m_byteCount),
        m_allocatedByteCount(that.m_allocatedByteCount)
{
    that.m_d = nullptr;
}

DeviceMemHandle::DeviceMemHandle(std::size_t byteCount)
{
    CU_CHECK(cudaMalloc(&m_d, byteCount));
    m_byteCount = m_allocatedByteCount = byteCount;
}

DeviceMemHandle::~DeviceMemHandle() {
    free();
}

DeviceMemHandle& DeviceMemHandle::operator=(DeviceMemHandle&& that)
{
    if (this != &that) {
        free();
        m_d = that.m_d;
        m_byteCount = that.m_byteCount;
        m_allocatedByteCount = that.m_allocatedByteCount;
        that.m_d = nullptr;
    }
    return *this;
}

void DeviceMemHandle::resize(std::size_t byteCount)
{
    if (m_allocatedByteCount < byteCount) {
        free();
        CU_CHECK(cudaMalloc(&m_d, byteCount));
        m_allocatedByteCount = m_byteCount = byteCount;
    }
    else if (m_allocatedByteCount >= byteCount) {
        m_byteCount = byteCount;
    }
}

std::size_t DeviceMemHandle::byteCount() const {
    return m_byteCount;
}

void *DeviceMemHandle::data() const {
    return m_d;
}

void DeviceMemHandle::clear() const {
    CU_CHECK(cudaMemset(m_d, 0, m_byteCount));
}

void DeviceMemHandle::upload(const void *src) const {
    CU_CHECK(cudaMemcpy(m_d, src, m_byteCount, cudaMemcpyHostToDevice));
}

void DeviceMemHandle::upload(const void *src, std::size_t byteCount, std::size_t dstStartIndex) const
{
    auto dst = static_cast<char*>(m_d) + dstStartIndex;
    CU_CHECK(cudaMemcpy(dst, src, byteCount, cudaMemcpyHostToDevice));
}

void DeviceMemHandle::download(void *dst) const {
    CU_CHECK(cudaMemcpy(dst, m_d, m_byteCount, cudaMemcpyDeviceToHost));
}

void DeviceMemHandle::download(void *dst, std::size_t byteCount, std::size_t srcStartIndex) const
{
    auto src = static_cast<char*>(m_d) + srcStartIndex;
    CU_CHECK(cudaMemcpy(dst, src, byteCount, cudaMemcpyDeviceToHost));
}

void DeviceMemHandle::free()
{
    if (m_d) {
        CU_CHECK(cudaFree(m_d));
        m_d = nullptr;
    }
}

} // gpu_ll

} // s3dmm
