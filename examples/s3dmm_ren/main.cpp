#define _USE_MATH_DEFINES

#include <QCoreApplication>
#include <QImage>
#include <EGL/egl.h>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <regex>
#include "opengl/App_MyVolumeRaycast.hpp"
#include "opengl/OpenGlSetup.hpp"
#include "saveFbImage.hpp"
#include "s3dmm/BlockTreeFieldService.hpp"
#include "s3dmm/ProgressReport.hpp"
#include "default_dense_field_gen.hpp"
#include <vlCore/VisualizationLibrary.hpp>
#include <vlCore/FileSystem.hpp>
#include <boost/optional.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace s3dmm;

using BT = BlockTree<3>;
using BlockId = typename BT::BlockId;
using BlockIndex = typename BT::BlockIndex;

class MyOpenGlContext : public vl::OpenGLContext
{
public:
    void swapBuffers() override {}
    void makeCurrent() override {}
    void update() override {
        dispatchUpdateEvent();
    }
};

struct CommandLineParameters
{
    string baseName;
    BlockIndex subtreeRootIndex;
    unsigned int subtreeRootLevel = 0;
    static CommandLineParameters parse(const QStringList& args)
    {
        CommandLineParameters result;
        if (args.size() > 1)
            result.baseName = args[1].toStdString();
        if (args.size() > 2) {
            auto subtreeIdString = args[2].toStdString();
            regex rx("^(\\d+),(\\d+),(\\d+),(\\d+)$");
            smatch m;
            if (!regex_match(subtreeIdString, m, rx))
                throw runtime_error("Invalid subtree block id");
            BOOST_ASSERT(m.size() == 2 + 3);
            for (auto i=0u; i<3; ++i)
                result.subtreeRootIndex[i] = boost::lexical_cast<unsigned int>(m[1+i]);
            result.subtreeRootLevel = boost::lexical_cast<unsigned int>(m[4]);
        }
        return result;
    }
};

class VisualizationData
{
public:
    explicit VisualizationData(
            const CommandLineParameters& param) :
        m_btfs(param.baseName),
        m_subtreeRoot(m_btfs.metadata().blockTree().blockAt(param.subtreeRootIndex, param.subtreeRootLevel)),
        m_interpolatorTimers(make_shared<DenseFieldInterpolator<3>::Timers>())
    {
        m_btfs.setInterpolatorTimers(m_interpolatorTimers);
    }

    void loadFrame(App_MyVolumeRaycast *applet)
    {
        ScopedTimerUser setDataTimerUser(&setDataTimer());
        applet->setData(
                    m_btfs.fieldName(m_fieldIndex),
                    m_fieldIndex,
                    m_timeFrame,
                    m_subtreeRoot);
        ++m_setDataInvocationCount;
    }

    ScopedTimer& setDataTimer() {
        if (!m_setDataTimer)
            m_setDataTimer = make_unique<ScopedTimer>();
        return *m_setDataTimer;
    }
    unsigned int setDataInvocationCount() const {
        return m_setDataInvocationCount;
    }

    void incTimeFrame() {
        if (m_btfs.timeFrameCount() > 0)
            m_timeFrame = (m_timeFrame + 1) % m_btfs.timeFrameCount();
    }

    BlockTreeFieldService<3>& blockTreeFieldService() {
        return m_btfs;
    }

    const BlockTreeFieldService<3>& blockTreeFieldService() const {
        return m_btfs;
    }

    shared_ptr<DenseFieldInterpolator<3>::Timers> interpolatorTimers() const {
        return m_interpolatorTimers;
    }

private:
    BlockTreeFieldService<3> m_btfs;
    BlockId m_subtreeRoot;

    unsigned int m_fieldIndex = 0;
    unsigned int m_timeFrame = 0;
    shared_ptr<DenseFieldInterpolator<3>::Timers> m_interpolatorTimers;
    mutable unique_ptr<ScopedTimer> m_setDataTimer;
    unsigned int m_setDataInvocationCount = 0;
};

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    try {
        using namespace vl;

        REPORT_PROGRESS_STAGES();

        auto param = CommandLineParameters::parse(app.arguments());
        shared_ptr<VisualizationData> visualizationData;
        if (!param.baseName.empty())
            visualizationData = make_shared<VisualizationData>(param);

        // Create EGL context
        auto width = 1024;
        auto height = 1024;
        OpenGlSetup oglSetup(width, height);

        // Initialize VL
        REPORT_PROGRESS_STAGE("Initialize VL");
        VisualizationLibrary::init();
        vl::defFileSystem()->directories().push_back(
            new vl::DiskDirectory((QCoreApplication::applicationDirPath() + "/data").toUtf8().data()));

        // Create applet and VL context
        REPORT_PROGRESS_STAGE("Create applet and VL context");
        auto volTexData = VolumeTextureDataInterface::newInstance(default_dense_field_gen);
        volTexData->setFieldService(&visualizationData->blockTreeFieldService());
        vl::ref<App_MyVolumeRaycast> applet = new App_MyVolumeRaycast(volTexData);

        applet->initialize();
        MyOpenGlContext ctx;
        ctx.addEventListener(applet.get());

        // Setup renderer
        REPORT_PROGRESS_STAGE("Setup renderer");
        auto fb = ctx.framebuffer();
        fb->setWidth(width);
        fb->setHeight(height);
        applet->rendering()->as<Rendering>()->renderer()->setFramebuffer( fb );
        applet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor( black );

        // Define the camera position and orientation
        vec3 eye    = vec3(0,10,35); // camera position
        vec3 center = vec3(0,0,0);   // point the camera is looking at
        vec3 up     = vec3(0,1,0);   // up direction
        mat4 view_mat = mat4::getLookAt(eye, center, up);
        auto camera = applet->rendering()->as<Rendering>()->camera();
        camera->setViewMatrix( view_mat );

        mat4 cameraRotation;
        cameraRotation.rotateXYZ(0, 1.f, 0);
        auto moveCamera = [&]() {
            eye = center + cameraRotation * (eye - center);
            view_mat = mat4::getLookAt(eye, center, up);
            camera->setViewMatrix( view_mat );
        };

        // Initialize VL context (that in turn initializes OpenGL functions)
        REPORT_PROGRESS_STAGE("Initialize VL context (that in turn initializes OpenGL functions)");
        ctx.initGLContext();

        // Generate applet initial events
        applet->resizeEvent(width, height);
        applet->initEvent();

        REPORT_PROGRESS_STAGE("Render scene and save images");
        auto IterationCount = 100;
        applet->setThresholdValue(0.f);
        applet->setSampleCount(1024);
        // applet->setRaycastMode(App_MyVolumeRaycast::MIP_Mode);
        // applet->setRaycastMode(App_MyVolumeRaycast::Isosurface_Mode);

        s3dmm::ScopedTimer loadFrameTimer;
        s3dmm::ScopedTimer drawSceneTimer;
        s3dmm::ScopedTimer saveImageTimer;

        for (auto iter=0; iter<IterationCount; ++iter) {
            // Load data, if any
            s3dmm::ScopedTimerUser loadFrameTimerUser(&loadFrameTimer);
            if (visualizationData) {
                if (iter == 0   ||   visualizationData->blockTreeFieldService().timeFrameCount() > 1) {
                    visualizationData->loadFrame(applet.get());
                    glFinish();
                }
            }
            loadFrameTimerUser.stop();

            // Draw scene
            s3dmm::ScopedTimerUser drawSceneTimerUser(&drawSceneTimer);
            applet->updateEvent();
            glFinish();
            drawSceneTimerUser.stop();

            // Save image and change the scene
            s3dmm::ScopedTimerUser saveImageTimerUser(&saveImageTimer);
            saveFbImage(width, height);
            moveCamera();
            applet->setThresholdValue(sin(static_cast<float>((iter+1)*M_PI/2/IterationCount)));
        }
        if (visualizationData) {
            auto& timers = *visualizationData->interpolatorTimers();
            cout << "Obtain block tree nodes average time: " << (timers.blockTreeNodesTimer.totalTime()/timers.invocationCount).format() << endl;
            cout << "Read sparse field average time: " << (timers.sparseFieldTimer.totalTime()/timers.invocationCount).format() << endl;
            cout << "Interpolate dense field average time: " << (timers.interpolateTimer.totalTime()/timers.invocationCount).format() << endl;
            cout << "Upload texture to GPU average time: " << (visualizationData->setDataTimer().totalTime()/visualizationData->setDataInvocationCount()).format() << endl;
        }
        cout << "Load frame average time: " << (loadFrameTimer.totalTime()/IterationCount).format() << endl;
        cout << "Draw scene average time: " << (drawSceneTimer.totalTime()/IterationCount).format() << endl;
        cout << "Save image time: " << (saveImageTimer.totalTime()/IterationCount).format() << endl;

        REPORT_PROGRESS_STAGE("Tear down");
        ctx.dispatchDestroyEvent();
        return EXIT_SUCCESS;
    }
    catch(exception& e) {
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}
