#include "s3dmm_cuda/selectGpu.hpp"
#include "cudaCheck.hpp"

namespace s3dmm {

void selectGpu(unsigned int gpuId)
{
    CU_CHECK(cudaSetDevice(gpuId));
}

}
