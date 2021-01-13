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

#include "MultiIndex.hpp"

#include <functional>
#include <mutex>
#include <vlCore/Time.hpp>
#include <vlGraphics/Applet.hpp>
#include <vlGraphics/Text.hpp>
#include <vlVolume/RaycastVolume.hpp>

#include "real_type.hpp"
#include "VolumeTextureDataInterface.hpp"
#include "VsWorkerInit.hpp"

class App_MyVolumeRaycast : public vl::Applet
{
public:
    using BlockId = s3dmm::detail::TreeBlockId<3>;

    App_MyVolumeRaycast(
            const std::shared_ptr<VolumeTextureDataInterface>& volTexData);

    std::shared_ptr<VolumeTextureDataInterface> volumeTextureData() const;

    vl::String appletInfo();

    void initEvent() override;
    void updateEvent() override;
    void updateScene() override;
    void mouseWheelEvent(int val) override;
    void keyPressEvent(unsigned short, vl::EKey key) override;
    void setData(
            const std::string& name,
            unsigned int fieldIndex, unsigned int timeFrame,
            const BlockId& subtreeRoot);
    void setChangeFieldCallback(std::function<void(int)> changeField);

    unsigned int sampleCount() const;
    void setSampleCount(unsigned int sampleCount);
    float thresholdValue() const;
    void setThresholdValue(float thresholdValue);

    void setIndex(
        unsigned int level, const s3dmm::MultiIndex<3, unsigned int>& index);

    enum RaycastMode
    {
        Isosurface_Mode,
        Isosurface_Transp_Mode,
        MIP_Mode,
        RaycastBrightnessControl_Mode,
        RaycastDensityControl_Mode,
        RaycastColorControl_Mode
    };

    RaycastMode raycastMode() const;
    void setRaycastMode(RaycastMode mode);

private:
    /* ----- raycast volume rendering options ----- */

    /* The sample step used to render the volume, the smaller the number
       the  better ( and slower ) the rendering will be. */
    float SAMPLE_STEP;

    /* volume visualization mode */
    RaycastMode MODE;

    /* If enabled, renders the volume using 3 animated lights. */
    bool DYNAMIC_LIGHTS;

    /* If enabled 3 colored lights are used to render the volume. */
    bool COLORED_LIGHTS;

    std::shared_ptr<VolumeTextureDataInterface> mVolTexData;

    void setupScene();

    /* visualize the given volume */
    void setupVolume();

    void updateVolumeTransform();

    void updateValThreshold(int val);

    void saveCameraPos() const;
    void loadCameraPos();

    vl::ref<vl::Transform> mVolumeTr;
    vl::mat4 mPrevVolumePos;
    vl::ref<vl::Transform> mLight0Tr;
    vl::ref<vl::Transform> mLight1Tr;
    vl::ref<vl::Transform> mLight2Tr;
    vl::ref<vl::Uniform> mValThreshold;
    vl::ref<vl::Light> mLight0;
    vl::ref<vl::Light> mLight1;
    vl::ref<vl::Light> mLight2;
    vl::ref<vl::GLSLProgram> mGLSL;
    vl::ref<vl::Actor> mVolumeAct;
    vl::ref<vl::RaycastVolume> mRaycastVolume;

    unsigned int mLevel = 0;
    s3dmm::MultiIndex<3, unsigned int> mIndex = {0, 0, 0};
    std::function<void(int)> mChangeField;

    using mutex_type = s3vs::VsWorkerInit::mutex_type;
    mutex_type& m_mut = s3vs::VsWorkerInit::getMutex();
};
