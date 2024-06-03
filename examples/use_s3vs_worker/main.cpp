#include "silver_bullets/iterate_struct/json_doc_converter.hpp"
#include "silver_bullets/iterate_struct/json_doc_io.hpp"
#include "silver_bullets/system/get_program_dir.hpp"
#include "silver_bullets/lib_warmup.hpp"
#include "silver_bullets/iterate_struct/PlainRepresentation.hpp"

#include "s3dmm/BlockTreeFieldService.hpp"
#include "s3vs/VsWorkerInterface.hpp"

#include "vlCore/Matrix4.hpp"
#include "vlCore/Vector3.hpp"

#include "iterate_struct_helpers/BoundingBoxPlainRepresentation.hpp"
#include "iterate_struct_helpers/iterateMultiIndex.hpp"
#include "iterate_struct_helpers/iterateVlVector.hpp"

#include <boost/program_options.hpp>

#include <QImage>

#include <filesystem>
#include <iostream>

using namespace std;

using vlVec3r = vl::Vector3<s3dmm::real_type>;

struct SubtreeSetDescriptionPlainRepresentation
{
    unsigned int level = 0;
    s3dmm::BoundingBoxPlainRepresentation<3, unsigned int> indexBox;

    static SubtreeSetDescriptionPlainRepresentation toPlain(const s3vs::SubtreeSetDescription& x) {
        return {
            x.level,
            silver_bullets::iterate_struct::toPlain(x.indexBox)
        };
    }
    static s3vs::SubtreeSetDescription fromPlain(const SubtreeSetDescriptionPlainRepresentation& x) {
        return {
            x.level,
            silver_bullets::iterate_struct::fromPlain<s3dmm::BoundingBox<3, unsigned int>>(x.indexBox)
        };
    }
};
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(
        SubtreeSetDescriptionPlainRepresentation, level, indexBox)

template <>
struct silver_bullets::iterate_struct::PlainRepresentation<s3vs::SubtreeSetDescription> {
    using type = SubtreeSetDescriptionPlainRepresentation;
};

struct AppOptions
{
    struct SceneState
    {
        struct CameraPosition
        {
            vlVec3r eye {0, 1, 5};
            vlVec3r center {0, 0, 0};
            vlVec3r up {0, 1, 0};
        };

        CameraPosition camera;

        unsigned int field = 0;
        unsigned int frame = 0;
        s3dmm::real_type threshold = s3dmm::make_real(0.5);
        s3vs::ColorTransferFunction colorTransferFunc = {
            {s3dmm::make_real(0.0), {1, 0, 1, 1}},
            {s3dmm::make_real(0.2), {0, 0, 1, 1}},
            {s3dmm::make_real(0.4), {0, 1, 1, 1}},
            {s3dmm::make_real(0.6), {0, 1, 0, 1}},
            {s3dmm::make_real(0.8), {1, 1, 0, 1}},
            {s3dmm::make_real(1.0), {1, 0, 0, 1}}
        };
    };

    std::string model;
    s3vs::Vec2i viewport {640, 480};
    SceneState scene;
    silver_bullets::iterate_struct::PlainRepresentation_t<s3vs::SubtreeSetDescription> blocks;
    s3dmm::real_type opacity = s3dmm::make_real(0.5);
    std::string output;

    static AppOptions exampleVal()
    {
        AppOptions opts;
        opts.model = "input-model-name";
        opts.output = "output-file-name";
        return opts;
    }
};

SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(s3vs::Block3Id, level, location);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(
    AppOptions::SceneState::CameraPosition, eye, center, up);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(AppOptions::SceneState, camera, field, frame, threshold, colorTransferFunc);
SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(AppOptions, model, viewport, scene, blocks, opacity, output);

SILVER_BULLETS_IMPORT_LIB_WARMUP_FUNC(s3vs_worker)

void run(int argc, char* argv[])
{
    SILVER_BULLETS_CALL_LIB_WARMUP_FUNC(s3vs_worker)

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

    s3vs::VsRenderSharedState st;
    st.shaderPath = filesystem::path(system::get_program_dir()) / "data";
    st.input.setViewportSize(opts.viewport);
    st.fieldSvc = make_shared<s3dmm::BlockTreeFieldService<3>>(opts.model);
    st.cameraTransform = s3vs::Matrix4r::getLookAt(
        opts.scene.camera.eye, opts.scene.camera.center, opts.scene.camera.up);
    st.input.setTimeFrame(opts.scene.frame);
    st.input.setPrimaryField(st.fieldSvc->fieldName(opts.scene.field));
    st.primaryFieldIndex = opts.scene.field;
    st.input.fieldAllParam().setIsosurfaceOpacity(opts.opacity);
    st.input.fieldAllParam().setIsosurfaceLevel(opts.scene.threshold);
    st.input.fieldAllParam().setColorTransferFunction(opts.scene.colorTransferFunc);

    auto vsWorker =
        silver_bullets::Factory<s3vs::VsWorkerInterface>::newInstance("Default");
    vsWorker->initialize(st.shaderPath, true, string(), 0);
    vsWorker->setRenderSharedState(&st);
    vsWorker->updateState(~0u);

    silver_bullets::sync::CancelController cc;

    auto blocks = silver_bullets::iterate_struct::fromPlain<s3vs::SubtreeSetDescription>(opts.blocks);
    auto imageRgba = vsWorker->renderScenePart(blocks, cc.checker());

    if (!opts.output.empty())
    {
        QImage image(
            imageRgba.image.bits.data(),
            imageRgba.image.size[0],
            imageRgba.image.size[1],
            QImage::Format_ARGB32);

        image.save(opts.output.c_str());
    }

    cout << "Image origin:\t(" << imageRgba.origin[0] << "; "
         << imageRgba.origin[1] << ")" << endl;
    cout << "Image size:\t(" << imageRgba.image.size[0] << "; "
         << imageRgba.image.size[1] << ")" << endl;
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
