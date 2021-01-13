#include <iostream>

#include "silver_bullets/fs_ns_workaround.hpp"
#include "silver_bullets/iterate_struct/json_doc_converter.hpp"
#include "silver_bullets/iterate_struct/json_doc_io.hpp"
#include "silver_bullets/system/get_program_dir.hpp"

#include "vlCore/Matrix4.hpp"
#include "vlCore/Vector3.hpp"

#include "s3vs/VsControllerInterface.hpp"
#include "s3vs/VsControllerFrameOutputRW.hpp"

#include "iterate_struct_helpers/iterateMultiIndex.hpp"
#include "iterate_struct_helpers/iterateVlVector.hpp"

#include <boost/program_options.hpp>

#include <boost/dll/shared_library.hpp>
#include <boost/system/system_error.hpp>

#include <thread>

#include <QImage>


using namespace std;

using vlVec3r = vl::Vector3<s3dmm::real_type>;

struct AppOptions
{
    struct SceneState
    {
        struct CameraPosition
        {
            vlVec3r eye;
            vlVec3r center;
            vlVec3r up;
        };

        CameraPosition camera;

        unsigned int field{0};
        unsigned int frame{0};
        s3dmm::real_type threshold = 0.5;
    };
    struct ComputingCaps
    {
        unsigned int compNodeCount{1};
        unsigned int gpuPerNodeCount{1};
        unsigned int cpuPerNodeCount{1};
        unsigned int workerThreadPerNodeCount{0};
    };
    struct ControllerOpts
    {
        bool measureRenderingTime{false};
    };

    std::string model;
    s3vs::Vec2i viewport;
    SceneState scene;
    s3dmm::real_type opacity = s3dmm::make_real(0.5);
    ComputingCaps computingCaps;
    ControllerOpts controllerOpts;

    std::string outputNamePrefix;
    bool writeFrames{false};

    static AppOptions exampleVal()
    {
        AppOptions opts{
            "input-model-name",
            {100, 200},
            {{vlVec3r(0, 10, 35), vlVec3r(0, 0, 0), vlVec3r(0, 1, 0)}, 0, 0},
            s3dmm::make_real(0.5),
            {1, 1, 1, 0},
            {false},
            "frame",
            false
        };
        return opts;
    }
};

SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(s3vs::Block3Id, level, location);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(
    AppOptions::SceneState::CameraPosition, eye, center, up);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(AppOptions::SceneState, camera, field, frame);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(
    AppOptions::ComputingCaps, compNodeCount, gpuPerNodeCount,
    cpuPerNodeCount, workerThreadPerNodeCount);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(AppOptions::ControllerOpts, measureRenderingTime);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(
    AppOptions, model, viewport, scene, opacity, computingCaps,
    controllerOpts, outputNamePrefix, writeFrames);

void loadLib(boost::dll::shared_library& lib, const string& name)
{
    if (!lib) {
        boost::system::error_code ec;
        lib.load(name, ec);
        if (ec)
            throw boost::system::system_error(ec);
    }
}

inline ostream& operator<<(ostream& s, const s3vs::Matrix4r& m)
{
    auto d = m.ptr();
    s << '[' << *d++;
    for (auto i=1; i<16; ++i)
        s << ", " << *d++;
    s << ']';
    return s;
}

struct M4 {
    explicit M4(const s3vs::Matrix4r& m) : m(m) {}
    s3vs::Matrix4r m;
};

ostream& operator<<(ostream& s, const M4& m4)
{
    auto& m = m4.m;
    string c[4][4];
    int w[4];
    for (auto j=0; j<4; ++j) {
        auto& wj = w[j];
        wj = 0;
        for (auto i=0; i<4; ++i) {
            c[i][j] = boost::lexical_cast<string>(m.e(i, j));
            wj = max(wj, static_cast<int>(c[i][j].size()));
        }
    }
    for (auto i=0; i<4; ++i) {
        for (auto j=0; j<4; ++j) {
            auto n = w[j] - static_cast<int>(c[i][j].size());
            for (auto k=0; k<n; ++k)
                cout << ' ';
            cout << c[i][j] << ' ';
        }
        cout << endl;
    }
    return s;
}

void run(int argc, char* argv[])
{
    using namespace silver_bullets;
    namespace po = boost::program_options;
    po::options_description desc;
    desc.add_options()(
        "config,c",
        po::value<std::string>(),
        "Path to JSON config")("example,e", "Show example config and exit")(
        "out_cfg,o",
        po::value<string>(),
        "Path to file where example config is to be saved");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.size() == 0)
    {
        cout << desc << endl;
        return;
    }

    AppOptions opts;

    if (vm.count("example"))
    {
        opts = AppOptions::exampleVal();
        auto json = iterate_struct::to_json_doc(opts);
        if (vm.count("out_cfg"))
        {
            auto name = vm["out_cfg"].as<string>();
            iterate_struct::write_json_doc(name, json);
            cout << "An example of configuration file is written in '"
                 << name << "'" << endl;
        }
        else
        {
            iterate_struct::write_json_doc(cout, json);
            cout << endl;
        }
        return;
    }

    auto cfgName = vm["config"].as<string>();
    auto json = iterate_struct::read_json_doc(cfgName);

    opts = iterate_struct::from_json_doc<decltype(opts)>(json);



    auto bindir = filesystem::path(system::get_program_dir());
    boost::dll::shared_library s3vs_lib;
    boost::dll::shared_library s3vs_worker;
    loadLib(s3vs_lib, bindir / "libs3vs_lib.so");
    loadLib(s3vs_worker, bindir / "libs3vs_worker.so");

    auto vsc = s3vs::VsControllerInterface::newInstance("default");

    {
        auto cca = vsc->computingCaps().access();
        cca->setCompNodeCount(opts.computingCaps.compNodeCount);
        cca->setCPUPerNodeNodeCount(opts.computingCaps.cpuPerNodeCount);
        cca->setGPUPerNodeNodeCount(opts.computingCaps.gpuPerNodeCount);
        cca->setWorkerThreadPerNodeNodeCount(opts.computingCaps.workerThreadPerNodeCount);
    }

    vsc->controllerOpts().access()->setMeasureRenderingTime(opts.controllerOpts.measureRenderingTime);

    vsc->setShaderPath(bindir / "data");

    vsc->start();

    auto input = vsc->input();
    {
        auto ia = input.access();
        ia->setProblemPath(opts.model);
        ia->setViewportSize(opts.viewport);
        auto fieldNames = vsc->fieldNames();
        if (fieldNames.size() <= opts.scene.field)
            throw std::runtime_error("Field index is out of range!");
        ia->setPrimaryField(fieldNames[opts.scene.field]);
        if (vsc->timeFrameCount() <= opts.scene.frame)
            throw std::runtime_error("Frame index is out of range!");
        ia->setTimeFrame(opts.scene.frame);
        ia->fieldAllParam().setIsosurfaceOpacity(opts.opacity);
        ia->fieldAllParam().setIsosurfaceLevel(opts.scene.threshold);
    }
    vsc->setCameraTransform(vl::Matrix4<s3dmm::real_type>::getLookAt(
                opts.scene.camera.eye, opts.scene.camera.center, opts.scene.camera.up));

    cout << "Current camera transformation:" << endl
         << vsc->cameraTransform() << endl;

    auto frameOutput = vsc->frameOutput();
    cout << "--- Frame output: ---" << endl
         << "shmid: " << frameOutput.shmem->shmid() << endl
         << "width: " << frameOutput.frameWidth << endl
         << "height: " << frameOutput.frameHeight << endl
         << "--- websock command line:" << endl
         << "websock -c " << frameOutput.frameWidth
         << " -r " << frameOutput.frameHeight
         << " -f 5"
         << " -s " << frameOutput.shmem->shmid()
         << endl << endl;

    cout << "Rotating camera each 100 ms... during next 1000 seconds" << endl;
    s3vs::MouseState ms;
    ms.flags = s3vs::MouseState::LeftButton;
    // ms.flags = s3vs::MouseState::MiddleButton;
    // ms.flags = s3vs::MouseState::ShiftKey;
    // ms.wheelDelta = 120;
    vsc->updateMouseState(ms);
    ++ms.x;

    auto W = frameOutput.frameWidth;
    auto H = frameOutput.frameHeight;
    s3vs::RgbaImage img { {W, H}, vector<unsigned char>(4*W*H) };

    double durationTotal = 0.;

    for (auto i=0; i<1000; ++i, ++ms.x) {
//        auto ct = vsc->cameraTransform();
//        cout << "---- " << i << " ----" << endl << M4(ct) << endl << endl;
        this_thread::sleep_for(chrono::milliseconds(300));
        vsc->updateMouseState(ms);

        if (opts.writeFrames || opts.controllerOpts.measureRenderingTime)
        {
            s3vs::VsControllerFrameOutputRW reader(frameOutput);
            s3vs::VsControllerFrameOutputHeader hdr;
            reader.readFrame(hdr, img.bits.data());
            if (opts.writeFrames)
            {
                QImage image(img.bits.data(), img.size[0], img.size[1], QImage::Format_ARGB32);
                string name = to_string(i);
                while (name.size() < 5)
                    name = "0" + name;
                name = opts.outputNamePrefix + name + ".png";
                image.save(name.c_str());
            }
            if (opts.controllerOpts.measureRenderingTime)
            {
                durationTotal += hdr.renderingDuration.durationMs;
                cout << "level = " << hdr.renderingDuration.level
                     << "\tduration(ms) = " << hdr.renderingDuration.durationMs
                     << endl;
            }
        }


    }

    if (opts.controllerOpts.measureRenderingTime)
    {
        cout << "Average duration(ms) = " << durationTotal/1000 << endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        run(argc, argv);
        return EXIT_SUCCESS;
    }
    catch (exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
