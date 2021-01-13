#include "s3dmm_cuda/DenseFieldInterpolator_cu_3.hpp"
#include "s3dmm_cuda/DeviceVector.hpp"
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include "progress_report_cu.hpp"
#include "sizeInMb.hpp"

namespace s3dmm {

struct I3
{
    static constexpr const unsigned int N = 3;
    BlockTreeNodeCoord m_d[N];

    I3() = default;

    __device__ I3(BlockTreeNodeCoord i0, BlockTreeNodeCoord i1, BlockTreeNodeCoord i2) {
        m_d[0] = i0;
        m_d[1] = i1;
        m_d[2] = i2;
    }

    __device__ unsigned int vertexIndexToOrdinal(unsigned int level) const
    {
        auto result = 0u;
        for (auto d=N-1; d!=~0u; --d)
            result = ((result << level)+result) + m_d[d];
        return result;
    }

};

inline constexpr unsigned int computeBlockCount(unsigned int size, unsigned int blockSize) {
    return (size+blockSize-1) / blockSize;
}

__global__ void scatterSparseFieldKernel(
        dfield_real *denseField,
        const real_type *sparseField,
        const I3 *n2i,
        unsigned int sparseFieldCount,
        unsigned int depth)
{
    unsigned int isparse = blockIdx.x * blockDim.x + threadIdx.x;
    if (isparse < sparseFieldCount) {
        auto idense = n2i[isparse].vertexIndexToOrdinal(depth);
        denseField[idense] = static_cast<dfield_real>(sparseField[isparse]);
    }
}

namespace interpolateDense3dFieldKernelConst
{
    constexpr unsigned int maxBlockSize1d = 8;
    constexpr auto Ns_log2 = 3;
    constexpr auto Ns = 1 << Ns_log2;   // Shared memory, dfield_real elements per thread
    constexpr unsigned int shmemSize = Ns*maxBlockSize1d*maxBlockSize1d*maxBlockSize1d;
    constexpr unsigned int d0 = Ns;                 // shmem address stride for next-x neighbor
    constexpr unsigned int d1 = d0*maxBlockSize1d;  // shmem address stride for next-y neighbor
    constexpr unsigned int d2 = d1*maxBlockSize1d;  // shmem address stride for next-z neighbor
    constexpr auto noFieldValue =
        make_dfield_real(BlockTreeFieldProvider<3>::noFieldValue());
    constexpr auto one_half = make_dfield_real(0.5);
    constexpr auto one_fourth = make_dfield_real(0.25);
    constexpr auto one_sixth = make_dfield_real(1./6);
}

__global__ void interpolateDense3dFieldKernel(
        dfield_real *denseField,
        unsigned int depth,
        unsigned int stage)
{
    using namespace interpolateDense3dFieldKernelConst;
    __shared__ dfield_real sh[shmemSize];
    auto g0 = threadIdx.x + (blockDim.x-1)*blockIdx.x;
    auto g1 = threadIdx.y + (blockDim.y-1)*blockIdx.y;
    auto g2 = threadIdx.z + (blockDim.z-1)*blockIdx.z;
    auto n = (1u << depth) + 1u;
    auto n3 = n*n*n;                        // Total size of denseField
    auto D = (1u << (depth-stage-1));       // Index stride for mid-edges, along edge
    auto D0 = D, D1 = n*D, D2 = n*D1;       // Address stride for mid-edges along dimensions 0, 1, 2
    auto Df0 = D1+D2, Df1 = D2+D0, Df2 = D0+D1; // Address stride for mid-faces prependicular to axes 0, 1, 2
    auto Dm = D0 + D1 + D2;                 // Address stride for mid-volume
    auto av = (g0*D0 + g1*D1 + g2*D2) << 1; // Index of this thread's vertex in denseField
    auto sv = threadIdx.x*d0 + threadIdx.y*d1 + threadIdx.z*d2; // Index of this thread's vertex in sh
    auto sv0 = (sv + d0)%shmemSize;         // Index of next-x thread's vertex in sh
    auto sv1 = (sv + d1)%shmemSize;         // Index of next-y thread's vertex in sh
    auto sv2 = (sv + d2)%shmemSize;         // Index of next-z thread's vertex in sh
    auto in0 = threadIdx.x+1 < blockDim.x;
    auto in1 = threadIdx.y+1 < blockDim.y;
    auto in2 = threadIdx.z+1 < blockDim.z;
    auto ghost0 = threadIdx.x+1 == blockDim.x   &&   blockIdx.x+1 < gridDim.x;
    auto ghost1 = threadIdx.y+1 == blockDim.y   &&   blockIdx.y+1 < gridDim.y;
    auto ghost2 = threadIdx.z+1 == blockDim.z   &&   blockIdx.z+1 < gridDim.z;

    auto load = [&](unsigned int s, unsigned int a) {
        if (a < n3)
            sh[s] = denseField[a];
        else
            sh[s] = NAN;
    };

    // Load vertex data into shared memory
    load(sv+0, av+0);
    load(sv+1, av+D0);
    load(sv+2, av+D1);
    load(sv+6, av+Df2);
    load(sv+3, av+D2);
    load(sv+5, av+Df1);
    load(sv+4, av+Df0);
    load(sv+7, av+Dm);
    __syncthreads();

    // Interpolate in the middle of 3 edges
    auto v = sh[sv];
    auto interpEdge = [&](unsigned int s, dfield_real v2) {
        if (isnan(sh[s])) {
            if (v == noFieldValue || v2 == noFieldValue)
                sh[s] = noFieldValue;
            else
                sh[s] = one_half*(v + v2);
        }
    };
    interpEdge(sv+1, sh[sv0]);
    interpEdge(sv+2, sh[sv1]);
    interpEdge(sv+3, sh[sv2]);
    __syncthreads();

    // Interpolate in the middle of 3 faces
    auto interpFace = [&](
            unsigned int s,
            dfield_real v1, dfield_real v2,
            dfield_real v3, dfield_real v4) {
        if (isnan(sh[s])) {
            if (v1 == noFieldValue || v2 == noFieldValue || v3 == noFieldValue || v4 == noFieldValue)
                sh[s] = noFieldValue;
            else
                sh[s] = one_fourth*(v1 + v2 + v3 + v4);
        }
    };
    auto v1 = sh[sv+1];
    auto v2 = sh[sv+2];
    auto v3 = sh[sv+3];
    interpFace(sv+4, v2, v3, sh[sv2+2], sh[sv1+3]);
    interpFace(sv+5, v1, v3, sh[sv2+1], sh[sv0+3]);
    interpFace(sv+6, v1, v2, sh[sv1+1], sh[sv0+2]);
    __syncthreads();

    // Interpolate in the middle of the volume
    auto interpVolume = [&](
            dfield_real v1, dfield_real v2,
            dfield_real v3, dfield_real v4,
            dfield_real v5, dfield_real v6)
    {
        auto s = sv + 7;
        if (isnan(sh[s])) {
            if (v1 == noFieldValue || v2 == noFieldValue || v3 == noFieldValue ||
                v4 == noFieldValue || v5 == noFieldValue || v6 == noFieldValue)
                sh[s] = noFieldValue;
            else
                sh[s] = one_sixth*(v1 + v2 + v3 + v4 + v5 + v6);
        }
    };
    interpVolume(sh[sv+4], sh[sv+5], sh[sv+6], sh[sv0+4], sh[sv1+5], sh[sv2+6]);

    // Write interpolated data back to global memory
    auto save = [&](unsigned int s, unsigned int a) {
        denseField[a] = sh[s];
    };

    // Write computed data back to global memory
    auto ig0 = g0 << (depth-stage);
    auto ig1 = g1 << (depth-stage);
    auto ig2 = g2 << (depth-stage);
    if (in0 && ig0+D<n && ig1<n && ig2<n && !(ghost1 || ghost2))
        save(sv+1, av+D0);
    if (in1 && ig0<n && ig1+D<n && ig2<n && !(ghost0 || ghost2))
        save(sv+2, av+D1);
    if (in0 && in1 && ig0+D<n && ig1+D<n && ig2<n && !ghost2)
        save(sv+6, av+Df2);
    if (in2 && ig0<n && ig1<n && ig2+D<n && !(ghost0 || ghost1))
        save(sv+3, av+D2);
    if (in0 && in2 && ig0+D<n && ig1<n && ig2+D<n && !ghost1)
        save(sv+5, av+Df1);
    if (in1 && in2 && ig0<n && ig1+D<n && ig2+D<n && !ghost0)
        save(sv+4, av+Df0);
    if (in0 && in1 && in2 && ig0+D<n && ig1+D<n && ig2+D<n)
        save(sv+7, av+Dm);
}

struct ReplaceNoFieldValueWithNan : thrust::unary_function<dfield_real, bool> {
    __host__ __device__ dfield_real operator()(const dfield_real& x) const {
        return x == interpolateDense3dFieldKernelConst::noFieldValue? NAN: x;
    }
};

template<>
void DenseFieldInterpolator_cu_3::interpolate(
        dfield_real *denseFieldDev,
        const std::vector<real_type>& sparseField,
        const BlockTreeNodes<N, BT>& subtreeNodes)
{
    CUDA_REPORT_PROGRESS_STAGES();

    CUDA_REPORT_PROGRESS_STAGE("Upload sparse field to GPU");
    // Fill dense fields with sparse field values, where it is defined,
    // and with NANs where the sparse field is undefined.
    auto depth = subtreeNodes.maxDepth();
    auto denseVertexCount = IndexTransform<N>::vertexCount(depth);

    thrust::fill(thrust::device, denseFieldDev, denseFieldDev+denseVertexCount, NAN);
    DeviceVector<I3> n2i(reinterpret_cast<const I3*>(subtreeNodes.data().n2i.data()), subtreeNodes.data().n2i.size());
    DeviceVector<real_type> devSparseField(sparseField);
    CUDA_REPORT_PROGRESS_IF_ENABLED(
                std::cout
                    << "Uploaded "
                    << (sizeInMb(subtreeNodes.data().n2i)+sizeInMb(sparseField))
                    << " MB to GPU" << std::endl);

    CUDA_REPORT_PROGRESS_STAGE("Scatter sparse field");
    constexpr unsigned int scatterThreadsPerBlock = 32;
    scatterSparseFieldKernel<<<
            computeBlockCount(n2i.size(), scatterThreadsPerBlock),
            scatterThreadsPerBlock>>>(
        denseFieldDev,
        devSparseField.data(),
        n2i.data(),
        n2i.size(),
        depth);

    CUDA_REPORT_PROGRESS_STAGE("Interpolate dense field");
    for (auto stage=0u; stage<depth; ++stage) {
        auto n = (1<<stage)+1;
        unsigned int blockSize1d, blockCount1d;
        if (n <= interpolateDense3dFieldKernelConst::maxBlockSize1d) {
            blockSize1d = n;
            blockCount1d = 1;
        }
        else {
            blockSize1d = interpolateDense3dFieldKernelConst::maxBlockSize1d;
            blockCount1d = computeBlockCount(n, blockSize1d-1);
        }
        dim3 blockSize(blockSize1d, blockSize1d, blockSize1d);
        dim3 blockCount(blockCount1d, blockCount1d, blockCount1d);
        interpolateDense3dFieldKernel<<<blockCount, blockSize>>>(denseFieldDev, depth, stage);
    }

    CUDA_REPORT_PROGRESS_STAGE("Replace noFieldValue with NAN in sparse field");
    thrust::transform(
                thrust::device,
                denseFieldDev, denseFieldDev+denseVertexCount,
                denseFieldDev, ReplaceNoFieldValueWithNan());
}

} // s3dmm
