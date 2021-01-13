#include "s3dmm_cuda/manage_profiling.hpp"

#include <cuda_profiler_api.h>

#include "cudaCheck.hpp"

namespace s3dmm {

void enableCudaProfiling() {
    CU_CHECK(cudaProfilerStart());
}

void disableCudaProfiling() {
    CU_CHECK(cudaProfilerStop());
}

}
