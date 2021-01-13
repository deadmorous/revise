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

#include "VolumeTextureDataBase.hpp"

using namespace std;
using namespace s3dmm;

class VolumeTextureCpuData :
        public VolumeTextureDataBase,
        public silver_bullets::FactoryMixin<VolumeTextureCpuData, VolumeTextureDataInterface>
{
public:
    vl::Texture *fieldTexture() const override
    {
        updateImages();
        if (!m_fieldTexValid) {
            m_fieldTexValid = true;
            textureFromImage(m_fieldTex, m_fieldImage.get());
        }
        return m_fieldTex.get();
    }

    vl::Texture *alphaTexture() const override
    {
        updateImages();
        if (!m_alphaTexValid) {
            m_alphaTexValid = true;
            textureFromImage(m_alphaTex, m_alphaImage.get());
        }
        return m_alphaTex.get();
    }

    Vec2<dfield_real> fieldRange() const override
    {
        updateImages();
        return m_fieldRange;
    }

private:

    void updateImages() const
    {
        if (needUpdate()) {
            setUpdated(true);
            computeImages();
        }
    }

    void computeImages() const
    {
        DFIELD_REPORT_PROGRESS_STAGES();
        using namespace vl;

        auto fs = fieldService();
        if (fs) {
            DFIELD_REPORT_PROGRESS_STAGE("Generate field (CPU)");
            auto& fid = fieldId();
            Vec2<dfield_real> unusedLocalFieldRange;
            fs->interpolate(
                unusedLocalFieldRange, m_denseField, fid.fieldIndex, fid.timeFrame, fid.subtreeRoot);
            m_fieldRange = fs->fieldRange(fieldId().fieldIndex).convertTo<dfield_real>();
        }
        else {
            DFIELD_REPORT_PROGRESS_STAGE("Generate default field");
            computeDefaultField(m_denseField);
            m_fieldRange = computeFieldRange(
                m_denseField,
                make_dfield_real(BlockTreeFieldProvider<3>::noFieldValue()));
        }

        DFIELD_REPORT_PROGRESS_STAGE("Allocate & fill textures");
        auto N = 1 << depth();
        if (!m_fieldImage || m_fieldImage->width() != N + 1)
        {
            m_fieldImage = new Image(
                N + 1, N + 1, N + 1, sizeof(float), IF_LUMINANCE, IT_FLOAT);
        }
        auto dataSize = static_cast<size_t>((N + 1) * (N + 1) * (N + 1));
        BOOST_ASSERT(dataSize == m_denseField.size());

        // Transform data to range [0, 1] and write it to the image; create alpha image
        auto scale = make_dfield_real(1) / (m_fieldRange[1] - m_fieldRange[0]);
        transform(
            m_denseField.begin(),
            m_denseField.end(),
            reinterpret_cast<float*>(m_fieldImage->pixels()),
            [&](const dfield_real& f) {
                if (isnan(f))
                    return 0.f;
                else
                    return static_cast<float>((f - m_fieldRange[0]) * scale);
            });

        if (!m_alphaImage || m_alphaImage->width() != N + 1)
        {
            m_alphaImage = new Image(
                N + 1, N + 1, N + 1, sizeof(float), IF_LUMINANCE, IT_FLOAT);
        }
        transform(
            m_denseField.begin(),
            m_denseField.end(),
            reinterpret_cast<float*>(m_alphaImage->pixels()),
            [&](const real_type& f) { return isnan(f) ? 0.0f : 1.0f; });

        m_fieldTexValid = false;
        m_alphaTexValid = false;
    }

    static void uploadTextureImage(vl::Texture* tex, const vl::Image* img)
    {
        BOOST_ASSERT(!!tex == !!img);
        if (img)
        {
            auto w = img->width();
            auto h = img->height();
            auto d = img->depth();
            glBindTexture(GL_TEXTURE_3D, tex->handle());
            VL_glTexImage3D(
                GL_TEXTURE_3D,
                0,
                tex->internalFormat(),
                w,
                h,
                d,
                0,
                img->format(),
                img->type(),
                img->pixels());
            glBindTexture(GL_TEXTURE_3D, 0);
            VL_CHECK_OGL()
        }
    }

    static void textureFromImage(vl::ref<vl::Texture>& tex, const vl::Image *img)
    {
        BOOST_ASSERT(img);
        if (tex)
            uploadTextureImage(tex.get(), img);
        else {
            tex =
                new vl::Texture(img, vl::TF_LUMINANCE16, false, false);
            tex->getTexParameter()->setMagFilter(vl::TPF_LINEAR);
            tex->getTexParameter()->setMinFilter(vl::TPF_LINEAR);
            tex->getTexParameter()->setWrap(vl::TPW_CLAMP_TO_EDGE);
        }
    }

    mutable vector<dfield_real> m_denseField;
    mutable vl::ref<vl::Image> m_fieldImage;
    mutable vl::ref<vl::Image> m_alphaImage;
    mutable vl::ref<vl::Texture> m_fieldTex;
    mutable bool m_fieldTexValid = false;
    mutable vl::ref<vl::Texture> m_alphaTex;
    mutable bool m_alphaTexValid = false;
    mutable Vec2<dfield_real> m_fieldRange;
};

SILVER_BULLETS_FACTORY_REGISTER_TYPE(VolumeTextureCpuData, "cpu");
