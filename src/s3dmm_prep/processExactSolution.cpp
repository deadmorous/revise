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

#include "processExactSolution.hpp"

#include "RunParameters.hpp"
#include "BoundingBox.hpp"

#include "silver_bullets/iterate_struct/json_doc_converter.hpp"
#include "silver_bullets/iterate_struct/json_doc_io.hpp"
#include "silver_bullets/iterate_struct/ConfigLoader.hpp"
#include "silver_bullets/fs_ns_workaround.hpp"
#include "silver_bullets/templatize/resolve_template_args.hpp"

#include "silver_bullets/task_engine/ParallelTaskScheduler.hpp"
#include "silver_bullets/task_engine/ThreadedTaskExecutor.hpp"
#include "silver_bullets/task_engine/SimpleTaskFunc.hpp"

#include "silver_bullets/factory.hpp"

#include "iterate_struct_helpers/BoundingBoxPlainRepresentation.hpp"
#include "iterate_struct_helpers/iterateMultiIndex.hpp"

#ifdef ENABLE_COLESO
#include "ColESo/coleso.h"
#else
#include "coleso_stub.h"
#endif // ENABLE_COLESO

#include "s3dmm/MetadataProvider.hpp"
#include "s3dmm/FullBlockTreeGenerator.hpp"
#include "s3dmm/BlockTreeExactFieldsProvider.hpp"
#include "s3dmm/BlockTreeFieldProvider.hpp"

#include "filename_util.hpp"

#include <mutex>
#include <numeric>
#include <list>

using namespace std;
using namespace s3dmm;
using namespace silver_bullets;

namespace {

template <unsigned int Nto, unsigned int Nfrom, class T>
ScalarOrMultiIndex_t<Nto, T> vToDim(const ScalarOrMultiIndex_t<Nfrom, T>& x)
{
    BOOST_STATIC_ASSERT(Nto <= Nfrom);
    ScalarOrMultiIndex_t<Nto, T> result;
    ScalarOrMultiIndex<Nto, T>::each_indexed(result, [&x](T& xi, unsigned int i) {
        xi = ScalarOrMultiIndex<Nfrom, T>::element(x, i);
    });
    return result;
}

template <unsigned int Nto, unsigned int Nfrom, class T>
BoundingBoxPlainRepresentation<Nto, T> bbToDim(const BoundingBoxPlainRepresentation<Nfrom, T>& x) {
    return { vToDim<Nto, Nfrom, T>(x.min), vToDim<Nto, Nfrom, T>(x.max) };
}



struct s_MultiscaleWaves : tPointFunction
{
    const char* description() const override {
        return "MultiscaleWaves";
    }

    tFuncType Type() const override {
        return FUNC_SCALAR;
    }

    void PointValue(double t, const double* coor, double* V) const override
    {
        auto x = coor[0];
        auto y = coor[1];
        auto z = coor[2];
        auto s = sin(0.5*M_PI*z);
        auto c = cos(0.5*M_PI*z);
        auto xx =  c*x + s*y;
        auto yy = -s*x + c*y;
        *V = sin(M_PI*xx/z) * sin(M_PI*yy/z) * sin(1/z + M_PI*t) - z;
    }

    void ReadParamsFromFile(const char* /*fname*/) override {
    }
};



template<unsigned int N>
struct ExactSolutionConfig
{
    string problemId;
    string parameterFileName;
    string outputDirectory;
    BoundingBoxPlainRepresentation<N, real_type> domain;
    ScalarOrMultiIndex_t<N, unsigned int> cell_count;
    real_type init_time;
    real_type time_step;
    unsigned int time_step_count;

    template<unsigned int Nto>
    ExactSolutionConfig<Nto> toDim() const {
        return {
            problemId,
            parameterFileName,
            outputDirectory,
            bbToDim<Nto>(domain),
            vToDim<Nto, N, unsigned int>(cell_count),
            init_time,
            time_step,
            time_step_count,
        };
    }
};

} // anonymous namespace

SILVER_BULLETS_DESCRIBE_TEMPLATE_STRUCTURE_FIELDS(
    ((unsigned int, N)), ExactSolutionConfig,
    problemId, parameterFileName, outputDirectory, domain, cell_count, init_time, time_step, time_step_count)

namespace {

using PointFuncFactory = Factory<tPointFunction>;

map<string, ExactSolutionConfig<3>> defaultExactConfigs = {{
    "4peak", {
        "4peak",                                // problemId
        "es_4peak.txt",                         // parameterFileName
        "data/4peak",                           // outputDirectory
        {{ -1, -1, -1 }, { 1, 1, 1 }},          // domain
        { 128, 128, 128 },                      // cell_count
        0,                                      // init_time
        0.01,                                   // time_step
        100                                     // time_step_count
    }}, {
    "PlanarGauss", {
        "PlanarGauss",                          // problemId
        "es_planargauss.txt",                   // parameterFileName
        "data/PlanarGauss",                     // outputDirectory
        {{ 43, -7, -7 }, { 57, 7, 7 }},         // domain
        { 128, 128, 128 },                      // cell_count
        25,                                     // init_time
        0.1,                                    // time_step
        100                                     // time_step_count
    }}, {
    "EntropyVortex", {
        "EntropyVortex",                        // problemId
        "es_entropyvortex.txt",                 // parameterFileName
        "data/EntropyVortex",                   // outputDirectory
        {{ 41, 37, -13.5 }, { 68, 64, 13.5 }},  // domain
        { 128, 128, 128 },                      // cell_count
        0,                                      // init_time
        0.1,                                    // time_step
        100                                     // time_step_count
    }}, {
    "PointSource",
    {
        "PointSource",                          // problemId
        "es_pointsource.txt",                   // parameterFileName
        "data/PointSource",                     // outputDirectory
        {{ 0, -50, -50 }, { 100, 50, 50 }},     // domain
        { 128, 128, 128 },                      // cell_count
        0,                                      // init_time
        1,                                      // time_step
        100                                     // time_step_count
    }}, {
    "RotatingDipole",
    {
        "RotatingDipole",                       // problemId
        "es_rotatingdipole.txt",                // parameterFileName
        "data/RotatingDipole",                  // outputDirectory
        // {{ -1, -1, -1 }, { 1.0078125, 1.0078125, 1.0078125 }},     // domain (solution is infinite at CS origin!)
        {{ -1, -1, -1 }, { 1, 1, 1 }},     // domain
        { 128, 128, 128 },                      // cell_count
        0,                                      // init_time
        1,                                      // time_step
        100                                     // time_step_count
    }}, {
    "MultiscaleWaves", {
        "MultiscaleWaves",
        "",                                     // parameterFileName - eny existing file is okay
        "data/MultiscaleWaves",                 // outputDirectory
        {{ -1, -1, 0.05 }, { 1, 1, 2.05 }},     // domain
        { 128, 128, 128 },                      // cell_count
        0,                                      // init_time
        0.02,                                   // time_step
        100                                     // time_step_count
    }}
};

template<unsigned int N>
ExactSolutionConfig<N> makedefaultExactSolutionConfig();

template<>
ExactSolutionConfig<1> makedefaultExactSolutionConfig<1>()
{
    ExactSolutionConfig<1> result;
    // TODO
    return result;
}

template<>
ExactSolutionConfig<2> makedefaultExactSolutionConfig<2>()
{
    ExactSolutionConfig<2> result;
    // TODO
    return result;
}

template<>
ExactSolutionConfig<3> makedefaultExactSolutionConfig<3>() {
    return defaultExactConfigs.at("PointSource");
}

#define REGISTER_POINT_FUNC(Name) \
    PointFuncFactory::registerType(#Name, []() { \
        return make_shared<s_##Name>(); \
    })

#define REGISTER_POINT_FUNC_D(Name) \
    PointFuncFactory::registerType(#Name, []() { \
        return make_shared<s_##Name<double>>(); \
    })

struct PointFunctionRegistrator {
    PointFunctionRegistrator() {
#ifdef ENABLE_COLESO
        REGISTER_POINT_FUNC  (4peak);
        REGISTER_POINT_FUNC  (PlanarGauss);
        REGISTER_POINT_FUNC  (PlanarSinus);
        REGISTER_POINT_FUNC  (EntropyVortex);
        REGISTER_POINT_FUNC_D(Source1D);
        REGISTER_POINT_FUNC  (AcousticShock);
        REGISTER_POINT_FUNC_D(Gaussian2D);
        REGISTER_POINT_FUNC  (Source2D);
        REGISTER_POINT_FUNC  (Gaussian3D);
        REGISTER_POINT_FUNC_D(Source3D);
        REGISTER_POINT_FUNC_D(PointSource);
        REGISTER_POINT_FUNC  (RotatingDipole);
        REGISTER_POINT_FUNC  (Coaxial);
        REGISTER_POINT_FUNC_D(Corner);
        REGISTER_POINT_FUNC  (CornerPlanar);
        REGISTER_POINT_FUNC  (Cylinder);
        REGISTER_POINT_FUNC  (SinusVisc);
        REGISTER_POINT_FUNC  (VortexInCylinder);
        REGISTER_POINT_FUNC_D(WaveInChannel);
        REGISTER_POINT_FUNC  (Riemann);
        REGISTER_POINT_FUNC  (SimpleWave);
        REGISTER_POINT_FUNC  (ViscShock);
        REGISTER_POINT_FUNC  (ConcCyl);
        REGISTER_POINT_FUNC  (PotentialCylinder);
        REGISTER_POINT_FUNC  (PotentialSphere);
        REGISTER_POINT_FUNC  (ViscSphere);
#endif // ENABLE_COLESO

        REGISTER_POINT_FUNC  (MultiscaleWaves);
    }
} pointFunctionRegistrator;

#undef REGISTER_POINT_FUNC

template<unsigned int N>
unsigned int blockTreeDepth(
    const RunParameters& param,
    const ExactSolutionConfig<N>& exactParam)
{
    using vector_type = typename BoundingBox<N,real_type>::vector_type;
    auto maxBlockTreeDepth = param.metadataMaxLevel + param.metadataBlockDepth;
    auto domain = BoundingBoxPlainRepresentation<N, real_type>::fromPlain(exactParam.domain);
    auto domainSize = domain.size();
    vector_type elementSizes;
    ScalarOrMultiIndex<N, real_type>::each_indexed(elementSizes, [&](real_type& si, unsigned int index) {
        si =
            ScalarOrMultiIndex<N, real_type>::element(domainSize, index) /
            ScalarOrMultiIndex<N, unsigned int>::element(exactParam.cell_count, index);
    });
    auto minElementSize = ScalarOrMultiIndex<N, real_type>::min(elementSizes);
    BoundingCube<N, real_type> pos(domain);
    auto depth = static_cast<unsigned int>(
        ceil(log(pos.size() / minElementSize) / log(make_real(2))) + make_real(0.1));
    if (depth > maxBlockTreeDepth)
        depth = maxBlockTreeDepth;
    return depth;
}

template<unsigned int N>
class DummyFieldGenerator
{
public:
    vector<string> fieldNames() const {
        return { "F", "x" };
    }
    unsigned int fieldCount() const {
        return 2;
    }
    void fieldValues(real_type *dst, const Vec<N, real_type>& pos) const
    {
        dst[0] = make_real(1);
        dst[1] = make_real(0);
        for (auto i=0u; i<N; ++i) {
            dst[0] *= sin(pos[i]);
            dst[1] += pos[i];
        }
    }
};

template<unsigned int N>
class ColesoFieldGenerator
{
public:
    explicit ColesoFieldGenerator(const shared_ptr<tPointFunction>& pointFunction) :
      m_pointFunction(pointFunction)
    {
    }

    vector<string> fieldNames() const
    {
        switch (m_pointFunction->Type()) {
            case FUNC_SCALAR:
                BOOST_ASSERT(m_pointFunction->NumVars() == 1);
                return { "f" };
            case FUNC_PULSATIONS:
                BOOST_ASSERT(m_pointFunction->NumVars() == 5);
                return { "drho", "du", "dv", "dw", "dp" };
            case FUNC_PHYSICAL:
                BOOST_ASSERT(m_pointFunction->NumVars() == 5);
                return { "rho", "u", "v", "w", "p" };
            case FUNC_CONSERVATIVE:
                BOOST_ASSERT(m_pointFunction->NumVars() == 5);
                return { "rho", "rho_x_u", "rho_x_v", "rho_x_w", "E" };
            case FUNC_TEMPVEL:
                BOOST_ASSERT(m_pointFunction->NumVars() == 5);
                return { "one", "u", "v", "w", "p_div_rho" };
            case FUNC_PULSATIONS_COMPLEX:
                BOOST_ASSERT(m_pointFunction->NumVars() == 10);
                return {
                    "Re_drho", "Re_duz", "Re_dur", "Re_duphi", "Re_dp",
                    "Im_drho", "Im_duz", "Im_dur", "Im_duphi", "Im_dp"
                };
            case FUNC_PULSCONS_COMPLEX:
                BOOST_ASSERT(m_pointFunction->NumVars() == 10);
                return {
                    "Re_rho", "Re_rho_x_u", "Re_rho_x_v", "Re_rho_x_w", "Re_E",
                    "Im_rho", "Im_rho_x_u", "Im_rho_x_v", "Im_rho_x_w", "Im_E",
                };
        }
        __builtin_unreachable();
    }

    unsigned int fieldCount() const {
        return m_pointFunction->NumVars();
    }

    void setTime(real_type time) {
        m_time = time;
    }

    real_type time() const {
        return m_time;
    }

    void fieldValues(real_type *dst, const Vec<N, real_type>& pos) const
    {
        auto pos_d = pos.template convertTo<double>();
        std::array<double, tPointFunction::NumVarsMax> dst_d;
        m_pointFunction->PointValue(static_cast<double>(m_time), pos_d.data(), dst_d.data());
        transform(dst_d.begin(), dst_d.begin() + m_pointFunction->NumVars(), dst, [](double x) -> real_type {
            return static_cast<real_type>(x);
        });
    }

private:
    shared_ptr<tPointFunction> m_pointFunction;
    real_type m_time = 0;
};

template<class T>
T parseElement(const string& text)
{
    istringstream s(text);
    T result;
    s >> result;
    if (s.fail() || (!s.eof() && static_cast<string::size_type>(s.tellg()) != text.size()))
        throw std::runtime_error("Failed to parse input string '" + text + "'");
    return result;
}

template<unsigned int N, class T>
MultiIndex<N, T> parseMultiIndexInitializer(const string& text)
{
    vector<string> elements;
    auto pos = text.find_first_of(',');
    if (pos == string::npos)
        return MultiIndex<N,T>::filled(parseElement<T>(text));
    else {
        MultiIndex<N,T> result;
        string::size_type start = 0;
        for (auto i=0u; i<N; ++i) {
            auto length = i+1 < N? pos-start: string::npos;
            result[i] = parseElement<T>(text.substr(start, length));
            start = pos + 1;
            if (i+2 < N) {
                pos = text.find_first_of(',', pos+1);
                if (pos == string::npos)
                    throw std::runtime_error("Failed to parse input string '" + text + "'");
            }
        }
        return result;
    }
}

} // anonymous namespace

template<unsigned int N>
void processExactSolutionTemplate(const RunParameters& param)
{
    using BT = BlockTree<N>;
    using BTN = BlockTreeNodes<N, BT>;
    using BlockId = typename BT::BlockId;
    using NodeIndex = typename BTN::NodeIndex;

    // Load configuration file, taking default parameters into account
    auto getDefaultParam = [](const string& problemId)
    {
        ExactSolutionConfig<3> result;
        auto defaultParamIt = defaultExactConfigs.find(problemId);
        if (defaultParamIt == defaultExactConfigs.end()) {
            result = makedefaultExactSolutionConfig<3>();
            if (!problemId.empty())
                result.problemId = problemId;
        }
        else
            result = defaultParamIt->second;
        if (!result.parameterFileName.empty())
            result.parameterFileName = "../../../dist/lib/ColESo/PARAMS/" + result.parameterFileName;
        return result;
    };

    ExactSolutionConfig<N> exactParam;
    if (param.exactConfigFileName.empty())
        exactParam = getDefaultParam(param.exactProblemId).template toDim<N>();
    else {
        iterate_struct::ConfigLoader cfgLoader(param.exactConfigFileName);
        exactParam = cfgLoader.value<ExactSolutionConfig<N>>();
        auto defaultProblemId = param.exactProblemId.empty()? "PointSource": param.exactProblemId;
        auto problemId = cfgLoader.valueAt<string>("/problemId", defaultProblemId);
        auto defaultParam = getDefaultParam(problemId);
        exactParam.parameterFileName = cfgLoader.valueAt("/parameterFileName", defaultParam.parameterFileName);
        exactParam.outputDirectory = cfgLoader.valueAt("/outputDirectory", defaultParam.outputDirectory);
        exactParam.domain          = cfgLoader.valueAt("/domain",          bbToDim<N>(defaultParam.domain));
        exactParam.cell_count      = cfgLoader.valueAt("/cell_count",      vToDim<N, 3, unsigned int>(defaultParam.cell_count));
        exactParam.init_time       = cfgLoader.valueAt("/init_time",       defaultParam.init_time);
        exactParam.time_step       = cfgLoader.valueAt("/time_step",       defaultParam.time_step);
        exactParam.time_step_count = cfgLoader.valueAt("/time_step_count", defaultParam.time_step_count);
    }
    auto hasParam = [](const string& param) {
        return !(param.empty() || param == "-");
    };
    if (hasParam(param.outputDirectory))
        exactParam.outputDirectory = param.outputDirectory;
    if (hasParam(param.exactCellCount))
        exactParam.cell_count = ScalarOrMultiIndex<N, unsigned int>::fromMultiIndex(
            parseMultiIndexInitializer<N, unsigned int>(param.exactCellCount));
    if (hasParam(param.exactTimeStepCount))
        exactParam.time_step_count = parseElement<unsigned int>(param.exactTimeStepCount);

    if (param.printExactConfig) {
        // Print config and return
        auto json = iterate_struct::to_json_doc(exactParam);
        iterate_struct::write_json_doc(cout, json);
        return;
    }

    // Create problem instance, initialize it, and allocate a vector for output variables
    auto pointFunc = PointFuncFactory::newInstance(exactParam.problemId);
    if (!exactParam.parameterFileName.empty()) {
        if (!filesystem::exists(exactParam.parameterFileName))
            throw runtime_error("Input file '" + exactParam.parameterFileName + "' does not exist");
        pointFunc->ReadParamsFromFile(exactParam.parameterFileName.c_str());
    }
    pointFunc->Init();

    // Resolve output directory
    if (filesystem::exists(exactParam.outputDirectory)) {
        if (!filesystem::is_directory(exactParam.outputDirectory))
            throw std::runtime_error("A file exists at output path '" +
                                     exactParam.outputDirectory +
                                     "' and it is not a directory");
    }
    else if (!filesystem::create_directories(exactParam.outputDirectory))
        throw std::runtime_error("Failed to create output directory '" + exactParam.outputDirectory + "'");

    // Resolve basic file names
    bool hasTimeSteps = exactParam.time_step_count > 1;
    auto primaryName = hasTimeSteps?
                           exactParam.outputDirectory + ".coleso":
                           exactParam.outputDirectory + "/data.coleso";
    auto mainMeshFileName = frameOutputFileName(primaryName, 0, hasTimeSteps);
    if (!filesystem::exists(mainMeshFileName)) {
        ofstream os(mainMeshFileName);
        os << "placeholder, do not delete";
    }
    auto metadataFileName = mainMeshFileName + ".s3dmm-meta";

    auto cbSubtreeReportGranularity = 1u;
    auto cbSubtreeInvocationCount = 0u;
    auto cbSubtreeNodes = [&](const auto& d) {
        if (cbSubtreeInvocationCount == 0)
            cout << "\nStarted generating subtree nodes" << endl;
        ++cbSubtreeInvocationCount;
        if (cbSubtreeInvocationCount % cbSubtreeReportGranularity == 0u) {
            auto& btn = d.blockTreeNodes;
            auto& btnd = btn.data();
            auto maxNodes = [](unsigned int depth){
                auto nodesPerEdge = (1u << depth) + 1u;
                return nodesPerEdge*nodesPerEdge*nodesPerEdge;
            }(btnd.maxDepth);
            auto actualNodes = btnd.n2i.size();
            auto byteCount = actualNodes*sizeof(NodeIndex) + sizeof(unsigned int) + sizeof(BlockId);
            cout << "subtree " << d.subtree+1 << " of " << d.subtreeCount
                 << ", level " << d.level
                 << ", fill " << static_cast<real_type>(actualNodes)/maxNodes
                 << ", size " << static_cast<real_type>(byteCount)/(1024*1024) << " MB"
#ifdef S3DMM_BLOCK_TREE_NODES_TIMING
                 << ", generation time " << btn.generationTime().wall*1000 << " ms"
#endif // S3DMM_BLOCK_TREE_NODES_TIMING
                 << endl;
        }
    };
    typename BlockTreeExactFieldsProvider<N>::ProgressCallback cbField;
    mutex coutMutex;
    if (!param.quiet) {
        cbField = [&](const auto& d) {
            lock_guard g(coutMutex);
            cout << "Generating field for block " << d.metadataBlock+1
                 << " of " << d.metadataBlockCount
                 << ", level " << d.level
                 << endl;
        };
    }



    REPORT_PROGRESS_STAGES();

    // Generate or read metadata
    REPORT_PROGRESS_STAGE("Generate metadata");
    auto depth = blockTreeDepth(param, exactParam);
    auto domain = BoundingBoxPlainRepresentation<N, real_type>::fromPlain(exactParam.domain);
    BoundingCube<N, real_type> pos(domain);
    using BTG = FullBlockTreeGenerator<N>;
    BTG btg(depth, pos);
    MetadataProvider<N, BTG> mp(
        metadataFileName,
        param.metadataBlockDepth,
        param.metadataMaxLevel,
        param.metadataMaxFullLevel,
        btg);
    if (!param.quiet)
        mp.setSubtreeNodesProgressCallback(cbSubtreeNodes);
    auto& md = mp.metadata();

    // Generate fields
    REPORT_PROGRESS_STAGE("Generate fields");
    shared_ptr<typename BlockTreeExactFieldsProvider<N>::Timers> fieldGenTimers;
    {
        namespace te = silver_bullets::task_engine;
        auto processTimestepFunc = te::makeSimpleTaskFunc([&](unsigned int frame) {
            auto meshFileName = frameOutputFileName(primaryName, frame, hasTimeSteps);
            {
                lock_guard g(coutMutex);
                cout << "Generating fields";
                if (hasTimeSteps)
                    cout << " at time step " << frame+1 << " of " << exactParam.time_step_count;
                cout << endl;
            }
            ColesoFieldGenerator<N> fieldGenerator(pointFunc);
            auto time = exactParam.init_time + frame * exactParam.time_step;
            fieldGenerator.setTime(time);
            BlockTreeExactFieldsProvider<N> btf(
                md,
                meshFileName,
                fieldGenerator,
                cbField,
                fieldGenTimers);

                cout << "Approximating fields at time step " << frame << endl;
        });

        using TaskFunc = te::SimpleTaskFunc;
        using TFR = te::TaskFuncRegistry<TaskFunc>;
        using TTX = te::ThreadedTaskExecutor<TaskFunc>;
        using PTS = te::ParallelTaskScheduler<TaskFunc>;

        PTS pts;
        constexpr auto resourceType = 0;
        constexpr auto processTimestepFuncId = 0;
        TFR taskFuncRegistry;
        taskFuncRegistry[processTimestepFuncId] = processTimestepFunc;
        for (auto ix=0u; ix<param.threadCount; ++ix)
            pts.addTaskExecutor(std::make_shared<TTX>(resourceType, &taskFuncRegistry));

        list<boost::any> taskInputs;
        list<boost::any*> taskInputPtrs;
        for (auto frame=0u; frame<exactParam.time_step_count; ++frame) {
            taskInputs.push_back(frame);
            taskInputPtrs.push_back(&taskInputs.back());
            te::const_pany_range input = { &taskInputPtrs.back(), &taskInputPtrs.back()+1 };
            te::pany_range output;
            pts.addTask({
                {1, 0, processTimestepFuncId, resourceType},
                output, input,
                std::function<void()>()
            });
        }
        pts.wait();
    }

    REPORT_PROGRESS_STAGE("Compute total field ranges");
    ColesoFieldGenerator<N> fieldGenerator(pointFunc);
    auto fieldNames = fieldGenerator.fieldNames();
    vector<unsigned int> fieldVariables(fieldNames.size());
    iota(fieldVariables.begin(), fieldVariables.end(), 0);
    vector<optional<Vec2<real_type>>> fieldRanges(fieldVariables.size());
    {
        cout << "Computing total field ranges" << endl;
        if (hasTimeSteps)
            cout << "Processing time step 0" << endl;
        cout << endl;

        auto addRange = [](optional<Vec2<real_type>>& dst, const Vec2<real_type>& range)
        {
            if (range[1] >= range[0]) {
                if (dst) {
                    auto& r = dst.value();
                    if (r[0] > range[0])
                        r[0] = range[0];
                    if (r[1] < range[1])
                        r[1] = range[1];
                }
                else
                    dst = range;
            }
        };

        foreach_byindex32(i, fieldVariables) {
            auto fieldIndex = fieldVariables[i];
            auto fieldName = fieldNames.at(fieldIndex);
            for (auto timeFrame=0u; timeFrame<exactParam.time_step_count; ++timeFrame) {
                auto fieldFileName =
                    frameOutputFileName(
                        primaryName, timeFrame, hasTimeSteps)
                     + ".s3dmm-field#" + fieldName;
                BlockTreeFieldProvider<N> btf(md, fieldFileName);
                for (auto& level : md.levels())
                    for (auto& block : level)
                        addRange(fieldRanges[i], btf.fieldRange(block.subtreeRoot()));
            }
        }
    }

    REPORT_PROGRESS_STAGE("Generating field info file");
    {
        string infoFileName;
        if (hasTimeSteps) {
            using namespace filesystem;
            auto s = splitFileName(primaryName);
            infoFileName = path(get<0>(s)).append(get<1>(s)).append(get<1>(s) + get<2>(s) + ".s3dmm-fields");
        }
        else
            infoFileName = primaryName + ".s3dmm-fields";
        ofstream os(infoFileName);
        if (os.fail())
            throw runtime_error(string("Failed to open output file '") + infoFileName + "'");
        os << "time_steps\n"
           << exactParam.time_step_count << endl << endl;
        os << "field\tmin\tmax\tanimated" << endl;
        foreach_byindex32(i, fieldVariables) {
            auto fieldIndex = fieldVariables[i];
            os << fieldNames[fieldIndex] << '\t';
            auto fieldRange = fieldRanges[i];
            if (fieldRange) {
                auto r = fieldRange.value();
                os << r[0] << '\t' << r[1];
            }
            else
                os << "-\t-";
            os << '\t' << (hasTimeSteps? 1: 0) << endl;
        }
    }
    REPORT_PROGRESS_END();

    if (fieldGenTimers) {
        auto timers = fieldGenTimers.get();
        cout << "******** Field approximation time ********" << endl;
        cout << "Field appoximation: init time = " << timers->fieldInitTimer.totalTime().format() << endl;
        cout << "Field appoximation: block tree nodes time = " << timers->blockTreeNodesTimer.totalTime().format() << endl;
        cout << "Field appoximation: reading time = " << timers->fieldReadTimer.totalTime().format() << endl;
        cout << "Field appoximation: propagation time = " << timers->fieldPropagationTimer.totalTime().format() << endl;
        cout << "Field appoximation: generation time = " << timers->fieldGenerationTimer.totalTime().format() << endl;
        cout << "Field appoximation: field transform time = " << timers->fieldTransformTimer.totalTime().format() << endl;
        cout << "Field appoximation: writing time = " << timers->fieldWriteTimer.totalTime().format() << endl;
        cout << "Field appoximation: other time = " << timers->otherOpTimer.totalTime().format() << endl;
        cout << "********" << endl << endl;
    }
}

struct callProcessExactSolutionTemplate {
    template<unsigned int N> void operator()(const RunParameters& param) const {
        processExactSolutionTemplate<N>(param);
    }
};

void processExactSolution(const RunParameters& param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Run the entire job");
    silver_bullets::resolve_template_args<
        integer_sequence<unsigned int, 1,2,3>>(
        make_tuple(param.spaceDimension), callProcessExactSolutionTemplate(), param);
}
