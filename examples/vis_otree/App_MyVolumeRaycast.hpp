#pragma once

#include <vlGraphics/Applet.hpp>
#include <vlCore/Time.hpp>
#include <vlVolume/RaycastVolume.hpp>
#include <vlGraphics/Text.hpp>
#include "real_type.hpp"
#include "MultiIndex.hpp"
#include <functional>

class App_MyVolumeRaycast : public vl::Applet
{
public:
    App_MyVolumeRaycast();

    vl::String appletInfo();

    void initEvent() override;
    void updateEvent() override;
    void updateScene() override;
    void mouseWheelEvent( int val ) override;
    void keyPressEvent(unsigned short, vl::EKey key) override;
    void setData(
            const std::string& name,
            const std::vector<s3dmm::real_type>& tex3d,
            const s3dmm::MultiIndex<3, unsigned int>& size);
    void setLinearTextureInterpolation(bool linear);
    void setChangeFieldCallback(std::function<void(int)> changeField);

    unsigned int sampleCount() const;
    void setSampleCount(unsigned int sampleCount);
    float thresholdValue() const;
    void setThresholdValue(float thresholdValue);

    enum RaycastMode {
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

    /* Use a separate 3d texture with a precomputed gradient to speedup the fragment shader.
       Requires more memory ( for the gradient texture ) but can speedup the rendering. */
    bool PRECOMPUTE_GRADIENT;

    bool mLinearTextureInterpolation = false;

    void setupScene();

    /* visualize the given volume */
    void setupVolume();

    void updateText();

    void updateValThreshold( int val );

    void saveCameraPos() const;
    void loadCameraPos();

    vl::ref<vl::Transform> mVolumeTr;
    vl::ref<vl::Transform> mLight0Tr;
    vl::ref<vl::Transform> mLight1Tr;
    vl::ref<vl::Transform> mLight2Tr;
    vl::ref<vl::Uniform> mValThreshold;
    vl::ref<vl::Text> mValThresholdText;
    vl::ref<vl::Light> mLight0;
    vl::ref<vl::Light> mLight1;
    vl::ref<vl::Light> mLight2;
    vl::ref<vl::GLSLProgram> mGLSL;
    vl::ref<vl::Actor> mVolumeAct;
    vl::ref<vl::RaycastVolume> mRaycastVolume;
    vl::ref<vl::Image> mVolumeFieldImage;
    vl::ref<vl::Image> mVolumeAlphaImage;
    vl::ref<vl::Texture> mVolFieldTex;
    vl::ref<vl::Texture> mVolAlphaTex;
    bool mImageDirty = true;
    vl::Time mFPSTimer;
    std::string mNname;
    std::function<void(int)> mChangeField;
};
