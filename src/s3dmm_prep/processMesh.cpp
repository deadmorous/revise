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

#include "processMesh.hpp"
#include "RunParameters.hpp"

#include "s3dmm/BlockTreeFromMesh.hpp"
#include "s3dmm/MetadataProvider.hpp"
#include "s3dmm/MeshDataProvider.hpp"
#include "s3dmm/MeshElementRefiner.hpp"
#include "s3dmm/BlockTreeIMappedFieldsFromFile.hpp"
#include "s3dmm/BlockTreeIFieldFromFile.hpp"
#include "s3dmm/BlockTreeFieldProvider.hpp"
#include "s3dmm/BlockTreeMappedFieldsProvider.hpp"
#include "s3dmm/BlockTreeFieldForMeshBoundary.hpp"

#include "filename_util.hpp"

#include "silver_bullets/templatize/resolve_template_args.hpp"

#include "silver_bullets/task_engine/ParallelTaskScheduler.hpp"
#include "silver_bullets/task_engine/ThreadedTaskExecutor.hpp"
#include "silver_bullets/task_engine/SimpleTaskFunc.hpp"
#include "silver_bullets/fs_ns_workaround.hpp"

#include <list>

using namespace std;

namespace {

void saveMeshBoundaryToTecplot(const s3dmm::MeshBoundaryExtractor& bx, const string& mainMeshFileName)
{
    using namespace s3dmm;
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Write boundary tecplot file");
    auto izone = 0;
    for (auto& zone : bx.zones()) {
        ostringstream oss;
        oss << mainMeshFileName + ".z-" << izone << "-boundary.tec";
        auto bfname = oss.str();
        switch (zone.elementType()) {
        case MeshElementType::Hexahedron: {
            auto& zd = zone.data<MeshElementType::Hexahedron>();
            ofstream os(bfname, ios::binary);
            if (os.fail())
                throw runtime_error(string("Failed to open output file '") + bfname + "'");
            cout << "Writing mesh boundary to file '" << bfname << "'" << endl;
            os << "variables = X Y Z" << endl;
            os << "zone  N = " << zd.nodes.size() << " E = " << zd.faces.size() << " DATAPACKING = POINT ZONETYPE = FEQUADRILATERAL" << endl;
            for (auto& node : zd.nodes)
                os << node[0] << '\t' << node[1] << '\t' << node[2] << endl;
            for (auto& fd : zd.faces) {
                auto face = fd.face;
                os << face[0]+1 << '\t' << face[1]+1 << '\t' << face[2]+1 << '\t' << face[3]+1 << endl;
            }
            break;
        }
        case MeshElementType::Quad: {
            auto& zd = zone.data<MeshElementType::Quad>();
            ofstream os(bfname, ios::binary);
            if (os.fail())
                throw runtime_error(string("Failed to open output file '") + bfname + "'");
            cout << "Writing mesh boundary to file '" << bfname << "'" << endl;
            os << "variables = X Y" << endl;
            os << "zone  N = " << zd.nodes.size() << " E = " << zd.faces.size() << " DATAPACKING = POINT ZONETYPE = FELINESEG" << endl;
            for (auto& node : zd.nodes)
                os << node[0] << '\t' << node[1] << endl;
            for (auto& fd : zd.faces) {
                auto& face = fd.face;
                os << face[0]+1 << '\t' << face[1]+1 << endl;
            }
            break;
        }
        default:
            BOOST_ASSERT(false);
            throw runtime_error("saveMeshBoundaryToTecplot() cannot deal with the current face element type");
        }
        ++izone;
    }
}

template <unsigned int N>
void processMeshTemplate(const RunParameters& param)
{
    using namespace s3dmm;
    using BT = BlockTree<N>;
    using BTN = BlockTreeNodes<N, BT>;
    using BlockId = typename BT::BlockId;
    using NodeIndex = typename BTN::NodeIndex;

    auto cbBlockTreeReportGranularity = 100000u;
    auto cbBlockTreeInvocationCount = 0u;
    auto cbBlockTree = [&](const auto& d) {
        if (cbBlockTreeInvocationCount == 0)
            cout << "Started generating BlockTree" << endl;
        ++cbBlockTreeInvocationCount;
        if (cbBlockTreeInvocationCount % cbBlockTreeReportGranularity == 0u)
            cout << "zone " << d.zone+1 << " of " << d.zoneCount
                 << ", element " << d.zoneElement+1 << " of " << d.zoneElementCount
                 << ", btsize " << d.blockTreeSize << endl;
    };

    auto cbSubtreeReportGranularity = 1u;
    auto cbSubtreeInvocationCount = 0u;
    auto cbSubtreeNodes = [&](const auto& d) {
        if (cbSubtreeInvocationCount == 0)
            cout << "\nStarted generating subtree nodes" << endl;
        ++cbSubtreeInvocationCount;
        if (cbSubtreeInvocationCount % cbSubtreeReportGranularity == 0u) {
            auto& btn = d.blockTreeNodes;
            auto& btnd = btn.data();
            auto maxNodes = IndexTransform<N>::vertexCount(btnd.maxDepth);
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
    typename BlockTreeFieldProvider<N>::ProgressCallback cbField;
    typename BlockTreeMappedFieldsProvider<N>::ProgressCallback cbMappedField;
    mutex coutMutex;
    if (!param.quiet) {
        cbField = [&](const auto& d) {
            lock_guard g(coutMutex);
            cout << "Generating field for block " << d.metadataBlock+1
                 << " of " << d.metadataBlockCount
                 << ", level " << d.level
                 << endl;
        };
        cbMappedField = [&](const auto& d) {
            lock_guard g(coutMutex);
            cout << "Generating "
                 << (d.stage == BlockTreeMappedFieldsProvider<N>::FieldMapStage? "map": "field")
                 << " for block " << d.metadataBlock+1
                 << " of " << d.metadataBlockCount
                 << ", level " << d.level
                 << endl;
        };

    }

    string mainMeshFileName;
    bool hasTimeSteps;
    tie(mainMeshFileName, hasTimeSteps) = firstOutputFrameFileName(param.meshFileName);

    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Generate metadata");
    MeshDataProvider mainMeshProvider(mainMeshFileName);
    MeshElementRefinerParam refinerParam(param.refinerParam);

    std::unique_ptr<MeshBoundaryExtractor> pmbx;
    auto mbxGetter = [&]() -> MeshBoundaryExtractor&
    {
        if (!pmbx) {
            if (param.extractBoundary) {
                REPORT_PROGRESS_STAGE("Extract mesh boundary");
                pmbx = std::make_unique<MeshBoundaryExtractor>(mainMeshProvider, refinerParam);
                REPORT_PROGRESS_STAGE("Extract mesh boundary");
            }
            else {
                MeshDataProvider noMeshProvider;
                pmbx = std::make_unique<MeshBoundaryExtractor>(noMeshProvider, refinerParam);
            }
        }
        return *pmbx;
    };

    using BTM = BlockTreeFromMesh<N, MeshDataProvider, MeshElementRefinerParam>;
    auto metadataFileName = mainMeshFileName + ".s3dmm-meta";
    BTM btm(metadataFileName,
            mainMeshProvider,
            param.metadataMaxFullLevel,
            param.metadataBlockDepth + param.metadataMaxLevel,
            refinerParam,
            mbxGetter,
            param.boundaryRefine);
    if (!param.quiet)
        btm.setBlockTreeProgressCallback(cbBlockTree);
    MetadataProvider<N, BTM> mp(
            metadataFileName,
            param.metadataBlockDepth,
            param.metadataMaxLevel,
            param.metadataMaxFullLevel,
            btm);
    if (!param.quiet)
        mp.setSubtreeNodesProgressCallback(cbSubtreeNodes);
    auto& md = mp.metadata();
    auto& bt = md.blockTree();

    if (param.extractBoundary) {
        REPORT_PROGRESS_STAGE("Approximate boundary");
        {
            BlockTreeFieldForMeshBoundary<N, MeshDataProvider, MeshElementRefinerParam> fieldForBoundary(
                md, mbxGetter, param.boundaryRefine);
            BlockTreeFieldProvider<N> shapeFieldProvider(
                        md,
                        mainMeshFileName + ".s3dmm-field#shape",
                        fieldForBoundary,
                        BlockTreeFieldGenerationPolicy::Separate,
                        cbField);
            if (param.saveBoundaryTecplot)
                saveMeshBoundaryToTecplot(mbxGetter(), mainMeshFileName);
        }
    }

    REPORT_PROGRESS_STAGE("Approximate fields");
    auto variableNames = mainMeshProvider.variables();
    auto fieldVariables = mainMeshProvider.fieldVariables();

    shared_ptr<typename BlockTreeFieldProvider<N>::Timers> fieldGenTimers;
    auto mappedFieldGenTimers = make_shared<typename BlockTreeMappedFieldsProvider<N>::Timers>();
    shared_ptr<typename BlockTreeIFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>::Timers> fieldGenInterpTimers;

    // Approximate field at first time step in the main thread;
    // that would lead to the generation of field map, suitable for further
    // use when generating fields at next time steps.
    cout << "Approximating fields";
    if (hasTimeSteps)
        cout << " at time step 0";
    cout << endl;
    BlockTreeMappedFieldsProvider<N> btf(
                md,
                mainMeshFileName,
                mainMeshFileName,
                BlockTreeIMappedFieldsFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
                    md, mainMeshProvider),
                cbMappedField, mappedFieldGenTimers);

    auto frame = 1u;
    if (hasTimeSteps) {
        namespace te = silver_bullets::task_engine;
        auto processTimestepFunc = te::makeSimpleTaskFunc([&](unsigned int frame) {
            auto meshFileName = frameOutputFileName(param.meshFileName, frame, true);
            BOOST_ASSERT(filesystem::exists(meshFileName));
            auto frameMeshProvider = make_unique<MeshDataProvider>(meshFileName);
            {
                lock_guard g(coutMutex);
                cout << "Approximating fields at time step " << frame << endl;
            }
            BlockTreeMappedFieldsProvider<N> btf(
                        md,
                        mainMeshFileName,
                        meshFileName,
                        BlockTreeIMappedFieldsFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
                            md, *frameMeshProvider),
                        cbMappedField, mappedFieldGenTimers);
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
        for (; ; ++frame) {
            auto meshFileName = frameOutputFileName(param.meshFileName, frame, true);
            if (filesystem::exists(meshFileName)) {
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
            else
                break;
        }
        pts.wait();
    }

    REPORT_PROGRESS_STAGE("Compute total field ranges");
    vector<optional<Vec2<real_type>>> fieldRanges(fieldVariables.size());
    {
        auto fieldNames = mainMeshProvider.variables();
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
            for (auto timeFrame=0u; timeFrame<frame; ++timeFrame) {
                auto fieldFileName = frameOutputFileName(param.meshFileName, timeFrame, hasTimeSteps) + ".s3dmm-field#" + fieldName;
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
            auto s = splitFileName(param.meshFileName);
            infoFileName = path(get<0>(s)).append(get<1>(s)).append(get<1>(s) + get<2>(s) + ".s3dmm-fields");
        }
        else
            infoFileName = param.meshFileName + ".s3dmm-fields";
        ofstream os(infoFileName);
        if (os.fail())
            throw runtime_error(string("Failed to open output file '") + infoFileName + "'");
        os << "time_steps\n"
           << frame << endl << endl;
        os << "field\tmin\tmax\tanimated" << endl;
        foreach_byindex32(i, fieldVariables) {
            auto fieldIndex = fieldVariables[i];
            os << variableNames[fieldIndex] << '\t';
            auto fieldRange = fieldRanges[i];
            if (fieldRange) {
                auto r = fieldRange.value();
                os << r[0] << '\t' << r[1];
            }
            else
                os << "-\t-";
            os << '\t' << (hasTimeSteps? 1: 0) << endl;
        }
        if (param.extractBoundary)
            os << "shape\t-1\t1\t0" << endl;
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

    if (mappedFieldGenTimers) {
        auto timers = mappedFieldGenTimers.get();
        cout << "******** Field approximation time ********" << endl;
        cout << "Field appoximation: init time = " << timers->fieldInitTimer.totalTime().format() << endl;
        cout << "Field appoximation: block tree nodes time = " << timers->blockTreeNodesTimer.totalTime().format() << endl;
        cout << "Field appoximation: reading time = " << timers->fieldReadTimer.totalTime().format() << endl;
        cout << "Field appoximation: propagation time = " << timers->fieldPropagationTimer.totalTime().format() << endl;
        cout << "Field appoximation: generation time = " << timers->fieldGenerationTimer.totalTime().format() << endl;
        cout << "Field appoximation: field transform time = " << timers->fieldTransformTimer.totalTime().format() << endl;
        cout << "Field appoximation: writing time = " << timers->fieldWriteTimer.totalTime().format() << endl;
        cout << "Field appoximation: map generation time = " << timers->fieldMapGenerationTimer.totalTime().format() << endl;
        cout << "Field appoximation: other time = " << timers->otherOpTimer.totalTime().format() << endl;
        cout << "********" << endl << endl;
    }

    if (fieldGenInterpTimers) {
        auto timers = fieldGenInterpTimers.get();
        cout << "******** Field approximation: interpolation operations time ********" << endl;
        cout << "Field appoximation (interp.): init time = " << timers->initTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): element nodes time = " << timers->elementNodesTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): element box time = " << timers->elementBoxTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): element box check time = " << timers->elementBoxCheckTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): element approx init time = " << timers->elementApproxInitTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): element approx run time = " << timers->elementApproxRunTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): field transform time = " << timers->fieldTransformTimer.totalTime().format() << endl;
        cout << "Field appoximation (interp.): other time = " << timers->otherOpTimer.totalTime().format() << endl;
        cout << "********" << endl << endl;
    }
}

struct callProcessMeshTemplate {
    template<unsigned int N> void operator()(const RunParameters& param) const {
        processMeshTemplate<N>(param);
    }
};

} // anonymous namespace

void processMesh(const RunParameters& param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Run the entire job");
    silver_bullets::resolve_template_args<
            integer_sequence<unsigned int, 1,2,3>>(
                make_tuple(param.spaceDimension), callProcessMeshTemplate(), param);
}
