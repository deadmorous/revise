#include "s3dmm_cuda/CudaGlResource.hpp"

#include "cudaCheck.hpp"

#include <cuda_gl_interop.h>

#include <boost/assert.hpp>

namespace s3dmm {
namespace gpu_ll {

namespace {

inline unsigned int toCudaGraphicsRegisterFlags(unsigned int cudaGlResourceUsageFlags)
{
    unsigned int result = 0u;
    if (cudaGlResourceUsageFlags & CudaGlResource::ReadOnly)
        result |= cudaGraphicsRegisterFlagsReadOnly;
    if (cudaGlResourceUsageFlags & CudaGlResource::WriteDiscard)
        result |= cudaGraphicsRegisterFlagsWriteDiscard;
    if (cudaGlResourceUsageFlags & CudaGlResource::SurfaceLoadStore)
        result |= cudaGraphicsRegisterFlagsSurfaceLoadStore;
    if (cudaGlResourceUsageFlags & CudaGlResource::TextureGather)
        result |= cudaGraphicsRegisterFlagsTextureGather;
    return result;
}

} // anonymous namespace

CudaGlResource::~CudaGlResource()
{
    unregister();
}

void CudaGlResource::registerTexture(GLuint texture, unsigned int usageFlags)
{
    if (m_glResource != texture) {
        unregister();
        if (texture) {
            CU_CHECK(cudaGraphicsGLRegisterImage(
                         reinterpret_cast<cudaGraphicsResource**>(&m_cudaResource),
                         texture,
                         GL_TEXTURE_3D,
                         toCudaGraphicsRegisterFlags(usageFlags)));
            BOOST_ASSERT(m_cudaResource);
        }
        m_glResource = texture;
        m_usageFlags = usageFlags;
    }
}

void CudaGlResource::unregister()
{
    if (m_cudaResource) {
        unmap();
        CU_CHECK(cudaGraphicsUnregisterResource(reinterpret_cast<cudaGraphicsResource_t>(m_cudaResource)));
        m_cudaResource = nullptr;
        m_glResource = 0;
        m_usageFlags = 0;
    }
}

void CudaGlResource::map()
{
    BOOST_ASSERT(m_cudaResource);
    if (!m_mapped) {
        CU_CHECK(cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&m_cudaResource)));
        m_mapped = true;
    }
}

void CudaGlResource::unmap()
{
    if (m_mapped) {
        CU_CHECK(cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&m_cudaResource)));
        m_mapped = false;
        m_surf = 0;
    }
}

void *CudaGlResource::mappedPtr()
{
    map();
    // void *ptr;
    // std::size_t size;
    // CU_CHECK(cudaGraphicsResourceGetMappedPointer(&ptr, &size, reinterpret_cast<cudaGraphicsResource_t>(m_cudaResource)));
    cudaArray_t array;
    CU_CHECK(cudaGraphicsSubResourceGetMappedArray(
                 &array,
                 reinterpret_cast<cudaGraphicsResource_t>(m_cudaResource),
                 0, 0));
    return array;
}

cudaSurfaceObject_t CudaGlResource::surface()
{
    BOOST_ASSERT(isRegistered());
    if (!m_surf) {
        cudaResourceDesc desc;
        memset(&desc, 0, sizeof(desc));
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = reinterpret_cast<cudaArray_t>(mappedPtr());
        CU_CHECK(cudaCreateSurfaceObject(&m_surf, &desc));
        BOOST_ASSERT(m_surf);
    }
    return m_surf;
}

} // gpu_ll
} // s3dmm
