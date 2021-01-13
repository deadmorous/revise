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
#include "s3dmm_cuda/DenseFieldInterpolator_cu_3.hpp"
#include "s3dmm_cuda/DenseFieldRenderHelper_cu.hpp"
#include "s3dmm_cuda/DeviceVector.hpp"

using namespace std;
using namespace s3dmm;

class VolumeTextureGpuData :
        public VolumeTextureGpuDataTpl<VolumeTextureGpuData>,
        public silver_bullets::FactoryMixin<VolumeTextureGpuData, VolumeTextureDataInterface>
{
public:
    VolumeTextureGpuData() :
        VolumeTextureGpuDataTpl<VolumeTextureGpuData>(this)
    {}

    Vec2<dfield_real> interpolate()
    {
        auto fs = fieldService();
        BOOST_ASSERT(fs);
        auto& fid = fieldId();
        m_field.resize(fs->denseFieldSize(fid.subtreeRoot));
        Vec2<dfield_real> fieldRange;
        fs->interpolateWith<DenseFieldInterpolator_cu_3>(
            fieldRange, m_field.data(), fid.fieldIndex, fid.timeFrame, fid.subtreeRoot);
        return fieldRange;
    }

    void uploadField(unsigned int depth, const dfield_real *dfield)
    {
        auto dfieldSize = IndexTransform<3>::vertexCount(depth);
        m_field.resize(dfieldSize);
        m_field.upload(dfield);
    }

    static unsigned int cudaTextureRegisterFlags() {
        return gpu_ll::CudaGlResource::WriteDiscard;
    }

    void prepareTextures(
            unsigned int depth,
            gpu_ll::CudaGlResource& fieldTexResource,
            gpu_ll::CudaGlResource& alphaTexResource,
            const Vec2<dfield_real>& fieldRange)
    {
        DenseFieldRenderHelper_cu::prepareTextures(
            depth, m_field.data(), fieldTexResource, alphaTexResource, fieldRange);
    }

private:
    DeviceVector<dfield_real> m_field;
};

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VolumeTextureGpuData, "gpu");

#endif // S3DMM_ENABLE_CUDA
