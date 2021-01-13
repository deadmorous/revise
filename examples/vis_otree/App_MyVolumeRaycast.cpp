#include "App_MyVolumeRaycast.hpp"

#include <vlVolume/VolumeUtils.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/FontManager.hpp>
#include <vlGraphics/GLSL.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlCore/VisualizationLibrary.hpp>

#include <boost/assert.hpp>

using namespace vl;
using namespace std;
using namespace s3dmm;

/*
  This example has been develop basing on the App_VolumeRaycast example.
*/
App_MyVolumeRaycast::App_MyVolumeRaycast() :
    SAMPLE_STEP(512.0f),
    MODE(Isosurface_Transp_Mode),
    DYNAMIC_LIGHTS(false),
    COLORED_LIGHTS(false),
    PRECOMPUTE_GRADIENT(false)
{
    mFPSTimer.start();
}

vl::String App_MyVolumeRaycast::appletInfo()
{
    return "Applet info: " + appletName() + "\n" +
        "Keys:\n" +
        "- Left/Right Arrow: change raycast technique.\n" +
        "- Up/Down Arrow: changes SAMPLE_STEP.\n" +
        "- L: toggles lights (useful only for isosurface).\n" +
        "- Mouse Wheel: change the bias used to render the volume.\n" +
        "\n" +
        "- Drop inside the window a set of 2D files or a DDS or DAT volume to display it.\n" +
        "\n";
}

void App_MyVolumeRaycast::initEvent()
{
    vl::Log::notify(appletInfo());

    if ( !Has_GLSL )
    {
        vl::Log::error( "OpenGL Shading Language not supported.\n" );
        vl::Time::sleep(2000);
        exit(1);
    }

    mLight0 = new Light;
    mLight1 = new Light;
    mLight2 = new Light;

    mLight0Tr = new Transform;
    mLight1Tr = new Transform;
    mLight2Tr = new Transform;
    rendering()->as<Rendering>()->transform()->addChild( mLight0Tr.get() );
    rendering()->as<Rendering>()->transform()->addChild( mLight1Tr.get() );
    rendering()->as<Rendering>()->transform()->addChild( mLight2Tr.get() );

    // volume transform
    mVolumeTr = new Transform;

    // val_threshold: manipulated via mouse wheel
    // - In Isosurface_Mode controls the iso-value of the isosurface
    // - In Isosurface_Transp_Mode controls the iso-value of the isosurface
    // - In MIP_Mode all the volume values less than this are discarded
    // - In RaycastBrightnessControl_Mode controls the brightness of the voxels
    // - In RaycastDensityControl_Mode controls the density of the voxels
    // - In RaycastColorControl_Mode controls the color-bias of the voxels
    mValThreshold = new Uniform( "val_threshold" );
    mValThreshold->setUniformF( 0.5f );

    // default volume image
    auto N = 256;
    auto L = 10.f;
    mVolumeFieldImage = new Image(N, N, N, sizeof(float), IF_LUMINANCE, IT_FLOAT);
    auto pixels = reinterpret_cast<float*>(mVolumeFieldImage->pixels());
    for (auto i=0; i<N; ++i) {
        auto x = static_cast<float>(i)/(N-1);
        for (auto j=0; j<N; ++j) {
            auto y = static_cast<float>(j)/(N-1);
            for (auto k=0; k<N; ++k) {
                auto z = static_cast<float>(k)/(N-1);
                auto v = 0.5f*(sin(L*(x-0.5f))*sin(L*(y-0.5f))*sin(L*(z-0.5f))+1);
                *pixels++ = v;
            }
        }
    }
    setupScene();
}

void App_MyVolumeRaycast::updateEvent()
{
    vl::Applet::updateEvent();

    if ( mFPSTimer.elapsed() > 1 )
    {
        mFPSTimer.start();
        openglContext()->setWindowTitle(
                    vl::Say("%s [%.1n] %s")
                    << mNname
                    << fps()
                    << appletName()  + " - " + vl::String("VL ") + vl::VisualizationLibrary::versionString() );
        // vl::Log::print( vl::Say("FPS=%.1n\n") << fps() );
    }
}

void App_MyVolumeRaycast::updateScene()
{
    if ( DYNAMIC_LIGHTS )
    {
        mat4 mat;
        // light 0 transform.
        mat = mat4::getRotation( Time::currentTime()*43, 0,1,0 ) * mat4::getTranslation( 20,20,20 );
        mLight0Tr->setLocalMatrix( mat );
        // light 1 transform.
        mat = mat4::getRotation( Time::currentTime()*47, 0,1,0 ) * mat4::getTranslation( -20,0,0 );
        mLight1Tr->setLocalMatrix( mat );
        // light 2 transform.
        mat = mat4::getRotation( Time::currentTime()*47, 0,1,0 ) * mat4::getTranslation( +20,0,0 );
        mLight2Tr->setLocalMatrix( mat );
    }
    if (mImageDirty) {
        if (!!mVolumeFieldImage == !!mVolFieldTex && !!mVolumeAlphaImage == !!mVolAlphaTex) {
            mImageDirty = false;
            auto uploadTextureImage = [](Texture *tex, Image *img) {
                BOOST_ASSERT(!!tex == !!img);
                if (img) {
                    auto w = img->width();
                    auto h = img->height();
                    auto d = img->depth();
                    glBindTexture(GL_TEXTURE_3D, tex->handle());
                    VL_glTexImage3D(GL_TEXTURE_3D, 0, tex->internalFormat(), w, h, d, 0, img->format(), img->type(), img->pixels());
                    glBindTexture(GL_TEXTURE_3D, 0);
                    VL_CHECK_OGL()
                }
            };
            uploadTextureImage(mVolFieldTex.get(), mVolumeFieldImage.get());
            uploadTextureImage(mVolAlphaTex.get(), mVolumeAlphaImage.get());
        }
        else
            setupVolume();
    }
}

void App_MyVolumeRaycast::mouseWheelEvent( int val )
{
    updateValThreshold( val );
}

void App_MyVolumeRaycast::keyPressEvent(unsigned short, vl::EKey key)
{
    static RaycastMode modes[] = { Isosurface_Mode, Isosurface_Transp_Mode, MIP_Mode, RaycastBrightnessControl_Mode, RaycastDensityControl_Mode, RaycastColorControl_Mode };
    static const auto ModeCount = sizeof(modes) / sizeof(modes[0]);

    auto sceneChanged = true;
    switch (key) {
    // left/right arrows change raycast technique
    case vl::Key_Right:
        MODE = modes[(MODE+1) % ModeCount];
        break;
    case vl::Key_Left:
        MODE = modes[(MODE+ModeCount-1) % ModeCount];
        break;

    // up/down changes SAMPLE_STEP
    case vl::Key_Up:
        SAMPLE_STEP += 64; // more precision
        break;
    case vl::Key_Down:
        SAMPLE_STEP -= 64; // less precision
        break;

    // L key toggles lights (useful only for isosurface)
    case vl::Key_L:
        if (!DYNAMIC_LIGHTS)
        {
            DYNAMIC_LIGHTS = true;
            COLORED_LIGHTS = false;
        }
        else if (DYNAMIC_LIGHTS && !COLORED_LIGHTS)
        {
            DYNAMIC_LIGHTS = true;
            COLORED_LIGHTS = true;
        }
        else
        {
            DYNAMIC_LIGHTS = false;
            COLORED_LIGHTS = false;
        }
        break;
    case vl::Key_X:
        saveCameraPos();
        sceneChanged = false;
        break;
    case vl::Key_Z:
        loadCameraPos();
        sceneChanged = false;
        break;
    case vl::Key_Period:
        if (mChangeField)
            mChangeField(1);
        sceneChanged = false;
        break;
    case vl::Key_Comma:
        if (mChangeField)
            mChangeField(-1);
        sceneChanged = false;
        break;
    default:
        sceneChanged = false;
    }

    if (sceneChanged)
        setupScene();
}

void App_MyVolumeRaycast::setData(
        const std::string& name,
        const std::vector<s3dmm::real_type>& tex3d,
        const s3dmm::MultiIndex<3, unsigned int>& size)
{
    mNname = name;

    if (!(mVolumeFieldImage->width() == size[0] &&
          mVolumeFieldImage->height() == size[1] &&
          mVolumeFieldImage->depth() == size[2]))
    {
        mVolumeFieldImage = new Image(size[0], size[1], size[2], sizeof(float), IF_LUMINANCE, IT_FLOAT);
    }
    auto dataSize = static_cast<size_t>(size[0]*size[1]*size[2]);
    BOOST_ASSERT(dataSize == tex3d.size());

    // Write data to the image
    transform(tex3d.begin(), tex3d.end(), reinterpret_cast<float*>(mVolumeFieldImage->pixels()), [&](const real_type& f) {
        return static_cast<float>(f);
    });

    mImageDirty = true;
}

void App_MyVolumeRaycast::setLinearTextureInterpolation(bool linear)
{
    mLinearTextureInterpolation = linear;
    mImageDirty = true;
}

void App_MyVolumeRaycast::setChangeFieldCallback(std::function<void(int)> changeField)
{
    mChangeField = changeField;
}

unsigned int App_MyVolumeRaycast::sampleCount() const {
    return static_cast<unsigned int>(SAMPLE_STEP);
}

void App_MyVolumeRaycast::setSampleCount(unsigned int sampleCount)
{
    SAMPLE_STEP = static_cast<float>(sampleCount);
    mGLSL->gocUniform( "sample_step" )->setUniformF( 1.0f / SAMPLE_STEP );
    updateText();
}

float App_MyVolumeRaycast::thresholdValue() const
{
    float val_threshold = 0.0f;
    mValThreshold->getUniform( &val_threshold );
    return val_threshold;
}

void App_MyVolumeRaycast::setThresholdValue(float thresholdValue)
{
    mValThreshold->setUniformF( clamp( thresholdValue, 0.0f, 1.0f ) );
    updateText();
}

auto App_MyVolumeRaycast::raycastMode() const -> RaycastMode {
    return MODE;
}

void App_MyVolumeRaycast::setRaycastMode(RaycastMode mode)
{
    MODE = mode;
    setupScene();
}

void App_MyVolumeRaycast::setupScene()
{
    // scrap previous scene
    sceneManager()->tree()->eraseAllChildren();
    sceneManager()->tree()->actors()->clear();
    mLight0->bindTransform( NULL );
    mLight1->bindTransform( NULL );
    mLight2->bindTransform( NULL );

    vl::ref<Effect> volume_fx = new Effect;
    // we don't necessarily need this:
    // volume_fx->shader()->enable( EN_BLEND );
    volume_fx->shader()->enable( EN_CULL_FACE );
    volume_fx->shader()->enable( EN_DEPTH_TEST );

    // NOTE
    // in these cases we render the back faces and raycast in back to front direction
    // in the other cases we render the front faces and raycast in front to back direction
    if ( MODE == RaycastBrightnessControl_Mode || MODE == RaycastDensityControl_Mode || MODE == RaycastColorControl_Mode )
    {
        volume_fx->shader()->enable( vl::EN_CULL_FACE );
        volume_fx->shader()->gocCullFace()->set( vl::PF_FRONT );
    }

    volume_fx->shader()->setRenderState( mLight0.get(), 0 );

    // light bulbs
    if ( DYNAMIC_LIGHTS )
    {
        // you can color the lights!
        if ( COLORED_LIGHTS )
        {
            mLight0->setAmbient( fvec4( 0.1f, 0.1f, 0.1f, 1.0f ) );
            mLight1->setAmbient( fvec4( 0.1f, 0.1f, 0.1f, 1.0f ) );
            mLight2->setAmbient( fvec4( 0.1f, 0.1f, 0.1f, 1.0f ) );
            mLight0->setDiffuse( vl::gold );
            mLight1->setDiffuse( vl::green );
            mLight2->setDiffuse( vl::royalblue );
        }

        // add the other two lights
        volume_fx->shader()->setRenderState( mLight1.get(), 1 );
        volume_fx->shader()->setRenderState( mLight2.get(), 2 );

        // animate the three lights
        mLight0->bindTransform( mLight0Tr.get() );
        mLight1->bindTransform( mLight1Tr.get() );
        mLight2->bindTransform( mLight2Tr.get() );

        // add also a light bulb actor
        vl::ref<Effect> fx_bulb = new Effect;
        fx_bulb->shader()->enable( EN_DEPTH_TEST );
        vl::ref<Geometry> light_bulb = vl::makeIcosphere( vec3( 0,0,0 ),1,1 );
        sceneManager()->tree()->addActor( light_bulb.get(), fx_bulb.get(), mLight0Tr.get() );
        sceneManager()->tree()->addActor( light_bulb.get(), fx_bulb.get(), mLight1Tr.get() );
        sceneManager()->tree()->addActor( light_bulb.get(), fx_bulb.get(), mLight2Tr.get() );
    }

    // the GLSL program that performs the actual raycasting
    mGLSL = volume_fx->shader()->gocGLSLProgram();
    mGLSL->gocUniform( "sample_step" )->setUniformF( 1.0f / SAMPLE_STEP );

    // attach vertex shader (common to all the raycasting techniques)
    mGLSL->attachShader( new GLSLVertexShader( "/glsl/volume_luminance_light.vs" ) );

    // attach fragment shader implementing the specific raycasting tecnique
    if ( MODE == Isosurface_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast_isosurface_alpha.fs" ) );
    else if ( MODE == Isosurface_Transp_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast_isosurface_transp_alpha.fs" ) );
    else if ( MODE == MIP_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast_mip.fs" ) );
    else if ( MODE == RaycastBrightnessControl_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast01.fs" ) );
    else if ( MODE == RaycastDensityControl_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast02.fs" ) );
    else if ( MODE == RaycastColorControl_Mode )
        mGLSL->attachShader( new GLSLFragmentShader( "/glsl/volume_raycast03.fs" ) );

    // manipulate volume transform with the trackball
    trackball()->setTransform( mVolumeTr.get() );

    // volume actor
    mVolumeAct = new Actor;
    mVolumeAct->setEffect( volume_fx.get() );
    mVolumeAct->setTransform( mVolumeTr.get() );
    sceneManager()->tree()->addActor( mVolumeAct.get() );
    // bind val_threshold uniform to the volume actor
    mVolumeAct->setUniform( mValThreshold.get() );

    // RaycastVolume will generate the actual actor's geometry upon setBox() invocation.
    // The geometry generated is actually a simple box with 3D texture coordinates.
    mRaycastVolume = new vl::RaycastVolume;
    mRaycastVolume->bindActor( mVolumeAct.get() );
    AABB volume_box( vec3( -10,-10,-10 ), vec3( +10,+10,+10 ) );
    mRaycastVolume->setBox( volume_box );

    // val_threshold text
    mValThresholdText = new Text;
    mValThresholdText->setFont( defFontManager()->acquireFont( "/font/bitstream-vera/VeraMono.ttf", 12 ) );
    mValThresholdText->setTextAlignment( TextAlignCenter );
    mValThresholdText->setAlignment( AlignHCenter | AlignBottom );
    mValThresholdText->setViewportAlignment( AlignHCenter | AlignBottom );
    mValThresholdText->translate( 0,5,0 );
    mValThresholdText->setBackgroundEnabled( true );
    mValThresholdText->setBackgroundColor( fvec4( 0,0,0,0.75 ) );
    mValThresholdText->setColor( vl::white );
    vl::ref<Effect> effect = new Effect;
    effect->shader()->enable( EN_BLEND );
    sceneManager()->tree()->addActor( mValThresholdText.get(), effect.get() );
    updateText();

    // let's visualize the volume!
    setupVolume();
}

void App_MyVolumeRaycast::setupVolume()
{
    mImageDirty = false;
    Effect* volume_fx = mVolumeAct->effect();

    volume_fx->shader()->enable( EN_BLEND );

    // for semplicity we don't distinguish between different image formats, i.e. IF_LUMINANCE, IF_RGBA etc.

    vl::ref<Image> gradient;
    if ( PRECOMPUTE_GRADIENT )
    {
        // note that this can take a while...
        gradient = vl::genGradientNormals( mVolumeFieldImage.get() );
    }

    Log::debug( Say("Volume: %n %n %n\n") << mVolumeFieldImage->width() << mVolumeFieldImage->height() << mVolumeFieldImage->depth() );

    // volume image field textue must be on sampler #0
    mVolFieldTex = new vl::Texture( mVolumeFieldImage.get(), TF_LUMINANCE16, false, false );
    volume_fx->shader()->gocTextureSampler( 0 )->setTexture( mVolFieldTex.get() );
    auto texInterpMode = mLinearTextureInterpolation? vl::TPF_LINEAR: vl::TPF_NEAREST;
    mVolFieldTex->getTexParameter()->setMagFilter( texInterpMode );
    mVolFieldTex->getTexParameter()->setMinFilter( texInterpMode );
    mVolFieldTex->getTexParameter()->setWrap( vl::TPW_CLAMP_TO_EDGE );
    volume_fx->shader()->gocUniform( "volume_texunit" )->setUniformI( 0 );
    mRaycastVolume->generateTextureCoordinates( ivec3( mVolumeFieldImage->width(), mVolumeFieldImage->height(), mVolumeFieldImage->depth() ) );

    if ( MODE == Isosurface_Mode || MODE == Isosurface_Transp_Mode ) {
        volume_fx->shader()->gocUniform( "has_alpha_texture" )->setUniformI( mVolumeAlphaImage ? 1 : 0 );
        if (mVolumeAlphaImage) {
            // volume image alpha textue must be on sampler #2
            mVolAlphaTex = new vl::Texture( mVolumeAlphaImage.get(), TF_LUMINANCE16, false, false );
            volume_fx->shader()->gocTextureSampler( 2 )->setTexture( mVolAlphaTex.get() );
            mVolAlphaTex->getTexParameter()->setMagFilter( vl::TPF_LINEAR );
            mVolAlphaTex->getTexParameter()->setMinFilter( vl::TPF_LINEAR );
            mVolAlphaTex->getTexParameter()->setWrap( vl::TPW_CLAMP_TO_EDGE );
            volume_fx->shader()->gocUniform( "volume_alpha_texunit" )->setUniformI( 2 );
        }
    }

    // generate a simple colored transfer function
    vl::ref<Image> trfunc;
    if ( COLORED_LIGHTS && DYNAMIC_LIGHTS ) {
        trfunc = vl::makeColorSpectrum( 128, vl::white, vl::white ); // let the lights color the volume
    }
    else {
        trfunc = vl::makeColorSpectrum( 128, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson );
    }

    // installs the transfer function as texture #1
    vl::ref< vl::Texture > trf_tex = new Texture( trfunc.get(), vl::TF_RGBA, false, false );
    trf_tex->getTexParameter()->setMagFilter( vl::TPF_LINEAR );
    trf_tex->getTexParameter()->setMinFilter( vl::TPF_LINEAR );
    trf_tex->getTexParameter()->setWrap( vl::TPW_CLAMP_TO_EDGE );
    volume_fx->shader()->gocTextureSampler( 1 )->setTexture( trf_tex.get() );
    volume_fx->shader()->gocUniform( "trfunc_texunit" )->setUniformI( 1 );

    // gradient computation, only use for isosurface methods
    if ( MODE == Isosurface_Mode || MODE == Isosurface_Transp_Mode )
    {
        volume_fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( PRECOMPUTE_GRADIENT ? 1 : 0 );
        if ( PRECOMPUTE_GRADIENT )
        {
            volume_fx->shader()->gocUniform( "gradient_texunit" )->setUniformI( 2 );
            vl::ref< vl::Texture > tex = new Texture( gradient.get(), TF_RGBA, false, false );
            tex->getTexParameter()->setMagFilter( vl::TPF_LINEAR );
            tex->getTexParameter()->setMinFilter( vl::TPF_LINEAR );
            tex->getTexParameter()->setWrap( vl::TPW_CLAMP_TO_EDGE );
            volume_fx->shader()->gocTextureSampler( 2 )->setTexture( tex.get() );
        }
    }

    // update text
    updateText();

    // refresh window
    openglContext()->update();
}

void App_MyVolumeRaycast::updateText()
{
    // update the val_threshold value and the raycast technique name
    String technique_name;
    switch(MODE)
    {
    case Isosurface_Mode: technique_name               = "raycast isosurface >"; break;
    case Isosurface_Transp_Mode: technique_name        = "< raycast transparent isosurface >"; break;
    case MIP_Mode: technique_name                      = "< raycast maximum intensity projection >"; break;
    case RaycastBrightnessControl_Mode: technique_name = "< raycast brightness control >"; break;
    case RaycastDensityControl_Mode: technique_name    = "< raycast density control >"; break;
    case RaycastColorControl_Mode: technique_name      = "< raycast color control"; break;
    }

    float val_threshold = 0;
    mValThreshold->getUniform( &val_threshold );
    mValThresholdText->setText( Say( "val_threshold = %n\n" "sample_step = 1.0 / %.0n\n" "%s" ) << val_threshold << SAMPLE_STEP << technique_name );
}

void App_MyVolumeRaycast::updateValThreshold( int val )
{
    float val_threshold = 0.0f;
    mValThreshold->getUniform( &val_threshold );
    val_threshold += val * 0.01f;
    val_threshold = clamp( val_threshold, 0.0f, 1.0f );
    mValThreshold->setUniformF( val_threshold );

    updateText();
    openglContext()->update();
}

void App_MyVolumeRaycast::saveCameraPos() const
{
    ofstream s("s3dmm_gui_camera_pos.bd", ios::binary);
    auto writeMatrix = [&s](const mat4& m) {
        s.write(reinterpret_cast<const char*>(m.ptr()), 16*sizeof(float));
    };
    writeMatrix(mVolumeTr->localMatrix());
    writeMatrix(rendering()->as<Rendering>()->camera()->viewMatrix());
}

void App_MyVolumeRaycast::loadCameraPos()
{
    ifstream s("s3dmm_gui_camera_pos.bd", ios::binary);
    if (!s.is_open())
        return;
    auto readMatrix = [&s](auto apply) {
        mat4 m;
        s.read(reinterpret_cast<char*>(m.ptr()), 16*sizeof(float));
        if (!s.fail())
            apply(m);
    };
    readMatrix([this](auto& m) {
        mVolumeTr->setLocalMatrix(m);
        mVolumeTr->computeWorldMatrix();
    });
    readMatrix([this](auto& m) {
        rendering()->as<Rendering>()->camera()->setViewMatrix(m);
    });
}
