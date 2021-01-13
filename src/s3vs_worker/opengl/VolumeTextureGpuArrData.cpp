/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

#ifdef S3DMM_ENABLE_CUDA

#include "VolumeTextureGpuDataTpl.hpp"
#include "s3dmm_cuda/DenseFieldInterpolatorArr_cu_3.hpp"
#include "s3dmm_cuda/DenseFieldRenderHelperArr_cu.hpp"
#include "s3dmm_cuda/Device3DArray.hpp"

using namespace std;
using namespace s3dmm;

class VolumeTextureGpuArrData :
        public VolumeTextureGpuDataTpl<VolumeTextureGpuArrData>,
        public silver_bullets::FactoryMixin<VolumeTextureGpuArrData, VolumeTextureDataInterface>
{
public:
    VolumeTextureGpuArrData() :
        VolumeTextureGpuDataTpl<VolumeTextureGpuArrData>(this)
    {}

    Vec2<dfield_real> interpolate()
    {
        auto fs = fieldService();
        BOOST_ASSERT(fs);
        auto& fid = fieldId();
        auto texelsPerEdge = fs->denseFieldVerticesPerEdge(fid.subtreeRoot);
        m_field.resize(texelsPerEdge, texelsPerEdge, texelsPerEdge,
                       {32, 0, 0, 0, CudaChannelFormat::Float});
        Vec2<dfield_real> fieldRange;
        fs->interpolateWith<DenseFieldInterpolatorArr_cu_3>(
            fieldRange, m_field, fid.fieldIndex, fid.timeFrame, fid.subtreeRoot);
        return fieldRange;
    }

    void uploadField(unsigned int depth, const dfield_real *dfield)
    {
        auto texelsPerEdge = IndexTransform<3>::template verticesPerEdge<BlockTreeNodeCoord>(depth);
        m_field.resize(texelsPerEdge, texelsPerEdge, texelsPerEdge,
                       {32, 0, 0, 0, CudaChannelFormat::Float});
        m_field.upload(dfield);
    }

    static unsigned int cudaTextureRegisterFlags() {
        return gpu_ll::CudaGlResource::SurfaceLoadStore;
    }

    void prepareTextures(
            unsigned int depth,
            gpu_ll::CudaGlResource& fieldTexResource,
            gpu_ll::CudaGlResource& alphaTexResource,
            const Vec2<dfield_real>& fieldRange) const
    {
        DenseFieldRenderHelperArr_cu::prepareTextures(
            depth, m_field, fieldTexResource, alphaTexResource, fieldRange);
    }

private:
    Device3DArray m_field;
};

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VolumeTextureGpuArrData, "gpu-a");

#endif // S3DMM_ENABLE_CUDA
