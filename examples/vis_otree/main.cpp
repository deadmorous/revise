#include <QApplication>

#include "s3dmm/Metadata.hpp"

#include <vlCore/VisualizationLibrary.hpp>
#include <vlQt5/Qt5Widget.hpp>
#include <vlCore/FileSystem.hpp>

#include <fstream>
#include <regex>

#include <boost/program_options.hpp>

#include "App_MyVolumeRaycast.hpp"

using namespace std;
using namespace s3dmm;

using MD = Metadata<3>;
using BT = MD::BT;
using BlockIndex = BT::BlockIndex;
using BlockId = BT::BlockId;

std::pair<unsigned int, BlockIndex> parseBlockIndex(const string& s)
{
    if (s.empty())
        return { 0, BlockIndex() };

    // Parse output subtree id
    regex rx("^(\\d+),(\\d+),(\\d+),(\\d+)$");
    smatch m;
    if (!regex_match(s, m, rx))
        throw runtime_error("Invalid subtree block id");
    BOOST_ASSERT(m.size() == 5);
    unsigned int n[4];
    transform(m.begin()+1, m.end(), n, [](const auto& x) {
        return boost::lexical_cast<unsigned int>(x);
    });
    return { n[0], {n[1], n[2], n[3]} };
}

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

struct Param
{
    string metadataFileName;
    unsigned int maxDepth = 0;
    string root;
    unsigned int blockDim = 3;
    bool linearInterp = true;
    bool levels = false;
    string bgColor = "000000";
    bool showRealDepth = true;
    bool compactView = false;
};

struct VlObjects
{
    vl::ref<App_MyVolumeRaycast> applet;
    vl::ref<vlQt5::Qt5Widget> qt5_window;
};

VlObjects initVisualization(const Param& param)
{
    VlObjects result;
    using namespace vl;

    /* init Visualization Library */
    VisualizationLibrary::init();
    vl::defFileSystem()->directories().push_back(
        new vl::DiskDirectory((QApplication::applicationDirPath() + "/data").toUtf8().data()));

    /* setup the OpenGL context format */
    OpenGLContextFormat format;
    format.setDoubleBuffer(true);
    format.setRGBABits( 8,8,8,0 );
    format.setDepthBufferBits(24);
    format.setStencilBufferBits(8);
    format.setFullscreen(false);
    //format.setMultisampleSamples(16);
    //format.setMultisample(true);

    /* create the applet to be run */
    result.applet = new App_MyVolumeRaycast;
    result.applet->initialize();
    /* create a native Qt5 window */
    result.qt5_window = new vlQt5::Qt5Widget;
    /* bind the applet so it receives all the GUI events related to the OpenGLContext */
    result.qt5_window->addEventListener(result.applet.get());
    /* target the window so we can render on it */
    result.applet->rendering()->as<Rendering>()->renderer()->setFramebuffer( result.qt5_window->framebuffer() );
    /* black background */
    result.applet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor(toColor(param.bgColor));
    /* define the camera position and orientation */
    vec3 eye    = vec3(0,10,35); // camera position
    vec3 center = vec3(0,0,0);   // point the camera is looking at
    vec3 up     = vec3(0,1,0);   // up direction
    mat4 view_mat = mat4::getLookAt(eye, center, up);
    result.applet->rendering()->as<Rendering>()->camera()->setViewMatrix( view_mat );
    /* Initialize the OpenGL context and window properties */
    int x = 10;
    int y = 10;
    int width = 512;
    int height= 512;
    result.qt5_window->initQt5Widget( "Volumetric data", format, nullptr, x, y, width, height );

    /* show the window */
    result.qt5_window->show();
    return result;
}

class VisualizationData
{
public:
    explicit VisualizationData(const Param& param) :
        m_param(param)
    {
        generateTexture();
    }

    void loadFrame(App_MyVolumeRaycast *applet)
    {
        auto size = texSizeForDepth(m_depth, m_param.compactView, true);
        BOOST_ASSERT(m_tex3d.size() == size*size*size);
        applet->setData(m_param.metadataFileName, m_tex3d, {size, size, size});
        applet->setLinearTextureInterpolation(m_param.linearInterp);
    }

private:
    Param m_param;
    vector<real_type> m_tex3d;
    unsigned int m_depth = 0;

    void generateTexture()
    {
        ifstream is(m_param.metadataFileName);
        if (is.fail())
            throw runtime_error(string("Failed to open input metadata file '") + m_param.metadataFileName + "'");

        MD m(is);
        if (m_param.levels) {
            BlockTree<3> bt;
            for (auto& level : m.levels()) {
                for (auto& block : level) {
                    bt.ensureBlockAt(block.blockIndex, block.level);
                }
            }
            generateTexture(bt);
        }
        else {
            generateTexture(m.blockTree());
        }
    }

    template<class BT>
    void generateTexture(const BT& bt)
    {
        auto rootPos = parseBlockIndex(m_param.root);
        auto root = bt.blockAt(rootPos.second, rootPos.first);

        // Compute the depth of the visualized part of the octree
        auto calcDepth = [&](const BlockId& root, unsigned int maxDepth) {
            auto result = 0u;
            bt.walk(root, [&](const BlockId& blockId) {
                auto relLevel = blockId.level - root.level;
                if (result < relLevel)
                    result = relLevel;
                return maxDepth == 0 || relLevel < maxDepth;
            });
            return result;
        };
        auto depth = calcDepth(root, m_param.maxDepth);
        auto totalDepth = m_param.showRealDepth && m_param.maxDepth != 0? calcDepth(root, 0): depth;

        // Compute texture size and reduce depth if necessary
        {
            const unsigned int MaxTexSize = 1000;   // Maybe increase it in the future
            auto limitedDepth = depth;
            while (texSizeForDepth(limitedDepth, m_param.compactView, true) > MaxTexSize)
                --limitedDepth;
            if (limitedDepth < depth) {
                cout << "WARNING: The requested depth " << depth << " has been reduced to " << limitedDepth << endl;
                depth = limitedDepth;
                if (!m_param.showRealDepth)
                    totalDepth = depth;
            }
        }

        // Generate 3D texture
        auto texSize = texSizeForDepth(depth, m_param.compactView, true);
        vector<real_type> tex3d(texSize*texSize*texSize, make_real(0));
        real_type fieldScale = totalDepth > 0?  make_real(1) / static_cast<real_type>(totalDepth): make_real(1);
        bt.walk(root, [&](const BlockId& blockId) {
            auto relLevel = blockId.level - root.level;
            BOOST_ASSERT(relLevel <= depth);
            auto size = texSizeForDepth(depth-relLevel, m_param.compactView, false);
            auto origin = blockOrigin(depth, blockId, root.level, m_param.compactView);
            BOOST_ASSERT(origin[0]+size <= texSize && origin[1]+size <= texSize && origin[2]+size <= texSize);
            auto levelToDisplay =
                    m_param.showRealDepth && relLevel == depth?
                        relLevel + calcDepth(blockId, 0):
                        relLevel;
            auto value = fieldScale * levelToDisplay;
            for (unsigned int i3=0; i3<size; ++i3) {
                auto b3 = i3 == 0 || i3 == size-1? 1u: 0u;
                for (unsigned int i2=0; i2<size; ++i2) {
                    auto b2 = i2 == 0 || i2 == size-1? 1u: 0u;
                    auto i0 = ((origin[2] + i3)*texSize + (origin[1] + i2))*texSize + origin[0];
                    for (unsigned int i1=0; i1<size; ++i1, ++i0) {
                        BOOST_ASSERT(i0 <= tex3d.size());
                        auto b1 = i1 == 0 || i1 == size-1? 1u: 0u;
                        auto b = b1 + b2 + b3;
                        if (m_param.blockDim == 4 ||
                            (m_param.compactView && relLevel == depth) ||
                            (m_param.blockDim == 3 && b == 0) ||
                            (m_param.blockDim < 3 && b >= 3u-m_param.blockDim))
                        {
                            BOOST_ASSERT(tex3d[i0] <= value);
                            tex3d[i0] = value;
                        }
                    }
                }
            }
            return relLevel < depth;
        });

        // Save data in member variables
        m_tex3d.swap(tex3d);
        m_depth = depth;
    }

    static unsigned int texSizeForDepth(unsigned int depth, bool compactView, bool includeMargins) {
        return compactView? (1u << depth) + (includeMargins? 2: 0): (1u << (depth+2)) + (1u << (depth)) - 2u;
    };

    static MultiIndex<3, unsigned int> blockOrigin(
            unsigned int depth,
            const BlockId& blockId,
            unsigned int rootLevel, bool compactView)
    {
        auto relLevel = blockId.level - rootLevel;
        if (compactView) {
            return ((blockId.location << (depth-relLevel)) & ((1 << depth)-1)) + MultiIndex<3, unsigned int>{1,1,1};
        }
        else {
            MultiIndex<3, unsigned int> result = {0, 0, 0};
            auto loc = blockId.location;
            MultiIndex<3, unsigned int> offset = {1, 1, 1};
            for (auto l=0u; l<relLevel; ++l, loc>>=1) {
                auto size = texSizeForDepth(depth-relLevel + l, compactView, false);
                result += (loc & 1) * size + offset;
            }
            return result;
        }
    }
};

int run(const Param& param)
{
    auto vlObjects = initVisualization(param);
    VisualizationData visualizationData(param);
    visualizationData.loadFrame(vlObjects.applet.get());

    /* run the Win32 message loop */
    int val = qApp->exec();

    /* deallocate the window with all the OpenGL resources before shutting down Visualization Library */
    vlObjects.qt5_window = nullptr;

    /* shutdown Visualization Library */
    vl::VisualizationLibrary::shutdown();

    return val;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    try {
        Param param;

        namespace po = boost::program_options;
        auto po_value = [](auto& x) {
            return po::value(&x)->default_value(x);
        };
        po::options_description po_generic("Gerneric options");
        po_generic.add_options()
                ("help,h", "produce help message");

        po::options_description po_main("Main parameters");
        po_main.add_options()
                ("input", po_value(param.metadataFileName), "Metadata file name (must contain 2D metadata)")
                ("depth", po_value(param.maxDepth), "Maximal depth for 3D texture")
                ("realdepth", po_value(param.showRealDepth), "Show real depth in leaves")
                ("root", po_value(param.root), "Subtree root (by default, the root of the block tree)")
                ("levels", po_value(param.levels), "Show only root nodes of metadata level structure")
                ("bdim", po_value(param.blockDim), "Visualized block dimension (0=vertices, 1=edges, 2=faces, 3=interior, 4-all)")
                ("lin", po_value(param.linearInterp), "Linear interpolation of 3D texture values")
                ("compact", po_value(param.compactView), "Compact view of tree nodes")
                ("bg", po_value(param.bgColor), "Background color");

        po::variables_map vm;
        auto po_alloptions = po::options_description()
                .add(po_generic).add(po_main);
        po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << po_alloptions << "\n";
            return 0;
        }

        return run(param);
    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
