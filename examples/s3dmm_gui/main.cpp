#include <QApplication>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/range/algorithm/find.hpp>
#include <regex>

#include "s3dmm/BlockTreeFieldService.hpp"
// #include "s3dmm/ProgressReport.hpp"

#include "default_dense_field_gen.hpp"

#include <vlCore/FileSystem.hpp>
#include <vlCore/VisualizationLibrary.hpp>
#include <vlQt5/Qt5Widget.hpp>

#include "opengl/App_MyVolumeRaycast.hpp"

using namespace std;
using namespace s3dmm;

using BT = BlockTree<3>;
using BlockId = typename BT::BlockId;
using BlockIndex = typename BT::BlockIndex;

struct CommandLineParameters
{
    string baseName;
    BlockIndex subtreeRootIndex;
    unsigned int subtreeRootLevel = 0;
    string fieldName;
    string bgColor = "000000";
    bool help = false;

    static CommandLineParameters parse(int argc, char* argv[])
    {
        namespace po = boost::program_options;

        auto po_value = [](auto& x) { return po::value(&x)->default_value(x); };

        CommandLineParameters result;

        po::options_description po_generic("Gerneric options");
        po_generic.add_options()("help,h", "Produce help message")(
            "quiet,q", "Don't display progress of separate operations");
        po::options_description po_basic("Main options");
        string root;
        po_basic.add_options()(
            "input", po::value(&result.baseName), "Mesh file name")(
            "root",
            po_value(root),
            "Subtree root (by default, the root of the block tree)")(
            "field", po_value(result.fieldName), "Name of field to display")(
            "bg", po_value(result.bgColor), "Background color");

        po::positional_options_description po_pos;
        po_pos.add("input", 1);

        po::variables_map vm;
        auto po_alloptions =
            po::options_description().add(po_generic).add(po_basic);
        po::store(
            po::command_line_parser(argc, argv)
                .options(po_alloptions)
                .positional(po_pos)
                .run(),
            vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << po_alloptions << "\n";
            result.help = true;
            return result;
        }

        if (result.baseName.empty())
            cerr << "WARNING: No mesh file name is specified, showing default field";

        if (!root.empty())
        {
            regex rx("^(\\d+),(\\d+),(\\d+),(\\d+)$");
            smatch m;
            if (!regex_match(root, m, rx))
                throw runtime_error("Invalid subtree block id");
            BOOST_ASSERT(m.size() == 2 + 3);
            for (auto i = 0u; i < 3; ++i)
                result.subtreeRootIndex[i] =
                    boost::lexical_cast<unsigned int>(m[1 + i]);
            result.subtreeRootLevel = boost::lexical_cast<unsigned int>(m[4]);
        }
        return result;
    }
};

struct VlObjects
{
    shared_ptr<VolumeTextureDataInterface> volTexData;
    vl::ref<App_MyVolumeRaycast> applet;
    vl::ref<vlQt5::Qt5Widget> qt5_window;
};

inline vl::fvec4 toColor(const string& color)
{
    if (color.empty())
        return vl::black;
    istringstream iss(color);
    unsigned int colorAsUint;
    iss >> hex >> colorAsUint;
    if (iss.fail())
        throw runtime_error("Invalid background color specification");
    return vl::makeColor((colorAsUint << 8) | 0xff);
}

VlObjects initVisualization(const CommandLineParameters& param)
{
    VlObjects result;
    using namespace vl;

    /* init Visualization Library */
    VisualizationLibrary::init();
    vl::defFileSystem()->directories().push_back(new vl::DiskDirectory(
        (QApplication::applicationDirPath() + "/data").toUtf8().data()));

    /* setup the OpenGL context format */
    OpenGLContextFormat format;
    format.setDoubleBuffer(true);
    format.setRGBABits(8, 8, 8, 0);
    format.setDepthBufferBits(24);
    format.setStencilBufferBits(8);
    format.setFullscreen(false);
    // format.setMultisampleSamples(16);
    // format.setMultisample(true);

    /* create the applet to be run */
    result.volTexData = VolumeTextureDataInterface::newInstance(default_dense_field_gen);
    result.applet = new App_MyVolumeRaycast(result.volTexData);
    result.applet->initialize();
    /* create a native Qt5 window */
    result.qt5_window = new vlQt5::Qt5Widget;
    /* bind the applet so it receives all the GUI events related to the
     * OpenGLContext */
    result.qt5_window->addEventListener(result.applet.get());
    /* target the window so we can render on it */
    result.applet->rendering()->as<Rendering>()->renderer()->setFramebuffer(
        result.qt5_window->framebuffer());
    /* black background */
    result.applet->rendering()
        ->as<Rendering>()
        ->camera()
        ->viewport()
        ->setClearColor(toColor(param.bgColor));
    /* define the camera position and orientation */
    vec3 eye = vec3(0, 10, 35);  // camera position
    vec3 center = vec3(0, 0, 0); // point the camera is looking at
    vec3 up = vec3(0, 1, 0);     // up direction
    mat4 view_mat = mat4::getLookAt(eye, center, up);
    result.applet->rendering()->as<Rendering>()->camera()->setViewMatrix(
        view_mat);
    /* Initialize the OpenGL context and window properties */
    int x = 10;
    int y = 10;
    int width = 512;
    int height = 512;
    result.qt5_window->initQt5Widget(
        "Volumetric data", format, nullptr, x, y, width, height);

    /* show the window */
    result.qt5_window->show();
    return result;
}

class VisualizationData
{
public:
    explicit VisualizationData(const CommandLineParameters& param) :
      m_btfs(param.baseName),
      m_subtreeRoot(m_btfs.metadata().blockTree().blockAt(
          param.subtreeRootIndex, param.subtreeRootLevel))
    {
        if (!param.fieldName.empty())
            m_fieldIndex = m_btfs.fieldIndex(param.fieldName);
    }

    void loadFrame(App_MyVolumeRaycast* applet)
    {
        applet->setIndex(m_subtreeRoot.level, m_subtreeRoot.location);
        applet->setData(
            m_btfs.fieldName(m_fieldIndex),
            m_fieldIndex,
            m_timeFrame,
            m_subtreeRoot);
    }

    void incTimeFrame()
    {
        if (m_btfs.timeFrameCount() > 0)
            m_timeFrame = (m_timeFrame + 1) % m_btfs.timeFrameCount();
    }

    BlockTreeFieldService<3>& blockTreeFieldService()
    {
        return m_btfs;
    }

    const BlockTreeFieldService<3>& blockTreeFieldService() const
    {
        return m_btfs;
    }

    unsigned int fieldIndex() const
    {
        return m_fieldIndex;
    }

    void setFieldIndex(unsigned int fieldIndex)
    {
        BOOST_ASSERT(fieldIndex < fieldCount());
        m_fieldIndex = fieldIndex;
    }

    unsigned int fieldCount() const
    {
        return m_btfs.fieldCount();
    }

private:
    BlockTreeFieldService<3> m_btfs;
    BlockId m_subtreeRoot;

    unsigned int m_fieldIndex = 0;
    unsigned int m_timeFrame = 0;
};

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    try
    {
        auto param = CommandLineParameters::parse(argc, argv);
        if (param.help)
            return EXIT_SUCCESS;
        auto vlObjects = initVisualization(param);
        std::shared_ptr<VisualizationData> visualizationData;

        if (!param.baseName.empty())
        {
            visualizationData = std::make_shared<VisualizationData>(param);
            vlObjects.volTexData->setFieldService(&visualizationData->blockTreeFieldService());
            visualizationData->loadFrame(vlObjects.applet.get());
            vlObjects.applet->setChangeFieldCallback([&](int delta) {
                auto fieldIndex = visualizationData->fieldIndex();
                auto fieldCount = visualizationData->fieldCount();
                fieldIndex = (fieldIndex + (delta > 0 ? 1 : fieldCount - 1))
                             % fieldCount;
                visualizationData->setFieldIndex(fieldIndex);
                visualizationData->loadFrame(vlObjects.applet.get());
            });

            if (visualizationData->blockTreeFieldService().timeFrameCount() > 1)
            {
                auto timer = new QTimer(&app);
                auto timeFrameInterval = 100;
                timer->setInterval(timeFrameInterval);
                QObject::connect(
                    timer,
                    &QTimer::timeout,
                    [&vlObjects, &visualizationData]() {
                        auto vd = visualizationData.get();
                        vd->incTimeFrame();
                        vd->loadFrame(vlObjects.applet.get());
                    });
                timer->start();
            }
        }

        /* run the Win32 message loop */
        int val = app.exec();

        /* deallocate the window with all the OpenGL resources before shutting
         * down Visualization Library */
        vlObjects.qt5_window = nullptr;

        /* shutdown Visualization Library */
        vl::VisualizationLibrary::shutdown();

        return val;
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
