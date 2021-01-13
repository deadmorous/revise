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

#pragma once

#include "VolumeTextureDataBase.hpp"
#include "s3dmm_cuda/CudaGlResource.hpp"

using namespace std;
using namespace s3dmm;

template<class Derived>
class VolumeTextureGpuDataTpl :
        public VolumeTextureDataBase
{
public:
    vl::Texture *fieldTexture() const override
    {
        updateTextures();
        return m_fieldTex.get();
    }

    vl::Texture *alphaTexture() const override
    {
        updateTextures();
        return m_alphaTex.get();
    }

    Vec2<dfield_real> fieldRange() const override
    {
        updateTextures();
        return m_fieldRange;
    }

protected:
    explicit VolumeTextureGpuDataTpl(Derived *derived) :
        m_derived(derived)
    {}

private:
    Derived *m_derived;

    void updateTextures() const
    {
        if (needUpdate()) {
            setUpdated(true);
            computeTextures();
        }
    }

    void computeTextures() const
    {
        DFIELD_REPORT_PROGRESS_STAGES();

        auto fs = fieldService();
        unsigned int currentDepth = depth();
        GLsizei texelsPerEdge = static_cast<GLsizei>(
            IndexTransform<3>::template verticesPerEdge<BlockTreeNodeCoord>(currentDepth));
        if (fs) {
            DFIELD_REPORT_PROGRESS_STAGE("Generate field (GPU)");
            m_derived->interpolate();
            m_fieldRange = fs->fieldRange(fieldId().fieldIndex).convertTo<dfield_real>();
        }
        else {
            DFIELD_REPORT_PROGRESS_STAGE("Generate default field");
            vector<dfield_real> denseFieldHost;
            computeDefaultField(denseFieldHost);
            m_fieldRange = computeFieldRange(
                denseFieldHost,
                make_dfield_real(BlockTreeFieldProvider<3>::noFieldValue()));
            BOOST_ASSERT(denseFieldHost.size() == s3dmm::IndexTransform<3>::vertexCount(currentDepth));
            m_derived->uploadField(DefaultFieldDepth, denseFieldHost.data());
        }
        DFIELD_REPORT_PROGRESS_STAGE("Allocate textures");
        alloc3dTexture(m_fieldTex, m_fieldTexResource, texelsPerEdge);
        alloc3dTexture(m_alphaTex, m_alphaTexResource, texelsPerEdge);

        DFIELD_REPORT_PROGRESS_STAGE("Fill textures");
        m_derived->prepareTextures(currentDepth, m_fieldTexResource, m_alphaTexResource, m_fieldRange);
    }

    static void alloc3dTexture(
            vl::ref<vl::Texture>& tex,
            gpu_ll::CudaGlResource& texResource,
            int texelsPerEdge)
    {
        if (!tex) {
            tex = new vl::Texture(texelsPerEdge, texelsPerEdge, texelsPerEdge, vl::TF_LUMINANCE16, false);
            tex->getTexParameter()->setMagFilter(vl::TPF_LINEAR);
            tex->getTexParameter()->setMinFilter(vl::TPF_LINEAR);
            tex->getTexParameter()->setWrap(vl::TPW_CLAMP_TO_EDGE);
        }
        else if (tex->width() != texelsPerEdge) {
            texResource.unregister();
            glBindTexture(GL_TEXTURE_3D, tex->handle());
            vl::VL_glTexImage3D(
                        GL_TEXTURE_3D, 0, vl::TF_LUMINANCE16,
                        texelsPerEdge, texelsPerEdge, texelsPerEdge, 0,
                        GL_LUMINANCE, GL_UNSIGNED_SHORT, nullptr);
            tex->setWidth(texelsPerEdge);
            tex->setHeight(texelsPerEdge);
            tex->setDepth(texelsPerEdge);
            glBindTexture(GL_TEXTURE_3D, 0);
        }
        texResource.registerTexture(tex->handle(), Derived::cudaTextureRegisterFlags());
    }

    mutable vl::ref<vl::Texture> m_fieldTex;
    mutable vl::ref<vl::Texture> m_alphaTex;
    mutable gpu_ll::CudaGlResource m_fieldTexResource;
    mutable gpu_ll::CudaGlResource m_alphaTexResource;
    mutable Vec2<dfield_real> m_fieldRange;
};
