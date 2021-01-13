#include "s3dmm_cuda/CudaChannelFormat.hpp"

#include <boost/static_assert.hpp>

struct cudaChannelFormatDesc;

namespace s3dmm {

cudaChannelFormatDesc CudaChannelFormat::makeCudaChannelFormatDesc() const
{
    return makeCudaChannelFormatDesc(*this);
}

cudaChannelFormatDesc CudaChannelFormat::makeCudaChannelFormatDesc(
    const CudaChannelFormat& channelFormat)
{
    BOOST_STATIC_ASSERT(cudaChannelFormatKindSigned == Signed);
    BOOST_STATIC_ASSERT(cudaChannelFormatKindUnsigned == Unsigned);
    BOOST_STATIC_ASSERT(cudaChannelFormatKindFloat == Float);
    BOOST_STATIC_ASSERT(cudaChannelFormatKindNone == None);
    return cudaCreateChannelDesc(
        channelFormat.x,
        channelFormat.y,
        channelFormat.z,
        channelFormat.w,
        static_cast<cudaChannelFormatKind>(channelFormat.f));
}

} // s3dmm
