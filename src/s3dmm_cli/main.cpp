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

#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <regex>

#include "s3dmm/BlockTree.hpp"
#include "s3dmm/BlockTreeNodes.hpp"
#include "s3dmm/MetadataProvider.hpp"

#include "s3dmm/BlockTreeFieldProvider.hpp"
#include "s3dmm/BlockTreeXWFieldFromFile.hpp"
#include "s3dmm/BlockTreeIFieldFromFile.hpp"

//#include "s3dmm/BlockTreeMappedFieldProvider.hpp"
//#include "s3dmm/BlockTreeIMappedFieldFromFile.hpp"
#include "s3dmm/BlockTreeMappedFieldsProvider.hpp"
#include "s3dmm/BlockTreeIMappedFieldsFromFile.hpp"

#include "s3dmm/BlockTree_io.hpp"
#include "s3dmm/BlockTreeNodes_io.hpp"
#include "s3dmm/Metadata.hpp"
#include "s3dmm/ProgressReport.hpp"
#include "s3dmm/MeshDataProvider.hpp"
#include "s3dmm/MeshElementRefiner.hpp"
#include "s3dmm/DenseFieldInterpolator.hpp"
// #include "s3dmm/MeshBoundaryExtractor.hpp"
#include "s3dmm/BlockTreeFieldForMeshBoundary.hpp"
#include "foreach_byindex32.hpp"
#include "filename_util.hpp"

#include <boost/program_options.hpp>

using namespace std;

namespace {

template< class T >
string toString(const T& x)
{
    ostringstream oss;
    oss << x;
    return oss.str();
}

template< class R >
string rangeToString(const R& r)
{
    ostringstream oss;
    auto i = 0u;
    for (auto it=r.begin(); it!=r.end(); ++it, ++i) {
        if (i > 0)
            oss << '\t';
        oss << toString(*it);
    }
    return oss.str();
}

template< class T >
string toString(const vector<T>& v) {
    return rangeToString(v);
}

template< class T >
string toString(const boost::iterator_range<T>& v) {
    return rangeToString(v);
}

template< class R >
void printRangeStart(const R& r, unsigned int maxElements = 10u)
{
    auto i = 0u;
    auto it=r.begin();
    for (; it!=r.end() && i<maxElements; ++it, ++i)
        cout << i << ":\t" << toString(*it) << endl;
    if (it != r.end()) {
        for (; it!=r.end(); ++it, ++i) {}
        cout << "... (" << i << " items in total)" << endl;
    }
    else if (i == 0)
        cout << "(empty)" << endl;
}

struct RunParameters
{
    string meshFileName;
    unsigned int metadataBlockDepth = 7;
    string outputFieldName;
    vector<unsigned int> outputBlockIndexInitializer;
    unsigned int outputBlockIndexLevel = ~0u;
    s3dmm::real_type refinerParam = s3dmm::make_real(1);
    unsigned int boundaryRefine = 1u;
    bool quiet = false;
    bool saveBoundaryTecplot = false;
    string fieldTecplotFileName(const string& fieldType, const string& fieldName) const {
        ostringstream oss;
        oss << meshFileName.substr(0, meshFileName.find_last_of('.'))
            << "_" << fieldType << "_"
            << fieldName << "-";
        foreach_byindex32(d, outputBlockIndexInitializer)
            oss << outputBlockIndexInitializer[d] << "-";
        oss << outputBlockIndexLevel
            << ".tec";
        return oss.str();
    }
    string denseFieldTecplotFileName(const string& fieldName) const {
        return fieldTecplotFileName("dense", fieldName);
    }
    string sparseFieldTecplotFileName(const string& fieldName) const {
        return fieldTecplotFileName("sparse", fieldName);
    }
};

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
                auto& face = fd.face;
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
void saveMeshBoundaryToPng(
        const typename s3dmm::Metadata<N>::BT&,
        const s3dmm::MeshBoundaryExtractor&,
        const string&)
{
    throw runtime_error(
                string("saveMeshBoundaryToPng() is not implemented for dimension ") +
                boost::lexical_cast<string>(N));
}

template <unsigned int N>
void run(const RunParameters& param)
{
    using namespace s3dmm;
    using BT = BlockTree<N>;
    using BTN = BlockTreeNodes<N, BT>;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using NodeIndex = typename BTN::NodeIndex;
    using Nvec = typename BT::vector_type;
    using BB = BoundingBox<N, real_type>;
    using MD = Metadata<N>;

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
    typename BlockTreeFieldProvider<N>::ProgressCallback cbField;
    typename BlockTreeMappedFieldsProvider<N>::ProgressCallback cbMappedField;
    if (!param.quiet) {
        cbField = [&](const auto& d) {
            cout << "Generating field for block " << d.metadataBlock+1
                 << " of " << d.metadataBlockCount
                 << ", level " << d.level
                 << endl;
        };
        cbMappedField = [&](const auto& d) {
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
            REPORT_PROGRESS_STAGE("Extract mesh boundary");
            pmbx = std::make_unique<MeshBoundaryExtractor>(mainMeshProvider, refinerParam);
            REPORT_PROGRESS_STAGE("Extract mesh boundary");
        }
        return *pmbx;
    };

    constexpr auto MetadataMaxLevel = 1000u;
    MetadataProvider<N, MeshDataProvider, MeshElementRefinerParam> mp(
                mainMeshFileName + ".s3dmm-meta",
                mainMeshProvider,
                param.metadataBlockDepth,
                MetadataMaxLevel,
                refinerParam,
                mbxGetter, param.boundaryRefine);
    if (!param.quiet) {
        mp.setBlockTreeProgressCallback(cbBlockTree);
        mp.setSubtreeNodesProgressCallback(cbSubtreeNodes);
    }
    auto& md = mp.metadata();
    auto& bt = md.blockTree();

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

    REPORT_PROGRESS_STAGE("Approximate fields");
    auto variableNames = mainMeshProvider.variables();
    auto fieldVariables = mainMeshProvider.fieldVariables();
    auto frame=0u;

    // auto fieldGenTimers = make_shared<typename BlockTreeFieldProvider<N>::Timers>();
    shared_ptr<typename BlockTreeFieldProvider<N>::Timers> fieldGenTimers;
    auto mappedFieldGenTimers = make_shared<typename BlockTreeMappedFieldsProvider<N>::Timers>();
    // auto fieldGenInterpTimers = make_shared<typename BlockTreeIFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>::Timers>();
    shared_ptr<typename BlockTreeIFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>::Timers> fieldGenInterpTimers;

    for (; frame==0 || hasTimeSteps; ++frame) {
        string meshFileName;
        unique_ptr<MeshDataProvider> frameMeshProvider;
        MeshDataProvider *meshProvider;
        if (frame == 0) {
            meshFileName = mainMeshFileName;
            meshProvider = &mainMeshProvider;
        }
        else {
            meshFileName = frameOutputFileName(param.meshFileName, frame, true);
            if (!experimental::filesystem::exists(meshFileName))
                break;
            frameMeshProvider = make_unique<MeshDataProvider>(meshFileName);
            meshProvider = frameMeshProvider.get();
        }
        cout << "Approximating fields";
        if (hasTimeSteps)
            cout << " at time step " << frame;
        cout << endl;
        BlockTreeMappedFieldsProvider<N> btf(
                    md,
                    mainMeshFileName,
                    meshFileName,
                    BlockTreeIMappedFieldsFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
                        md, *meshProvider),
                    cbMappedField, mappedFieldGenTimers);

//        for (auto fieldIndex : fieldVariables) {
//            cout << "Approximating field " << variableNames[fieldIndex];
//            if (hasTimeSteps)
//                cout << " at time step " << frame;
//            cout << endl;
////            BlockTreeFieldProvider<N> btf(
////                        md,
////                        meshFileName + ".s3dmm-field#" + variableNames[fieldIndex],
////                        //BlockTreeXWFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
////                        //    md, fieldIndex, *meshProvider, refinerParam),
////                        // BlockTreeFieldGenerationPolicy::Separate,
////                        BlockTreeIFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
////                            md, fieldIndex, *meshProvider, fieldGenInterpTimers),
////                        BlockTreeFieldGenerationPolicy::Propagate,
////                        cbField, fieldGenTimers);
//            BlockTreeMappedFieldProvider<N> btf(
//                        md,
//                        meshFileName,
//                        meshFileName + ".s3dmm-field#" + variableNames[fieldIndex],
//                        BlockTreeIMappedFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
//                            md, fieldIndex, *meshProvider),
//                        cbMappedField, mappedFieldGenTimers);
//        }
    }

    {
        string infoFileName;
        if (hasTimeSteps) {
            using namespace experimental::filesystem;
            auto s = splitFileName(param.meshFileName);
            infoFileName = path(get<0>(s)).append(get<1>(s)).append(get<1>(s) + get<2>(s) + ".s3dmm-fields");
        }
        else
            infoFileName = param.meshFileName + ".s3dmm-fields";
        ofstream os(infoFileName);
        if (os.fail())
            throw runtime_error(string("Failed to open output file '") + infoFileName + "'");
        foreach_byindex32(i, fieldVariables) {
            if (i > 0)
                os << " ";
            auto fieldIndex = fieldVariables[i];
            os << variableNames[fieldIndex];
        }
        os << " shape";
        os << endl;
        os << frame << endl;
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

    if (fieldVariables.empty())
        cout << "There are no field variables!" << endl;
    else {
        auto fieldName = param.outputFieldName.empty()? variableNames[fieldVariables[1]]: param.outputFieldName;
        BlockTreeFieldProvider<N> btf(
                md,
                mainMeshFileName + ".s3dmm-field#" + fieldName,
                BlockTreeXWFieldFromFile<N, MeshDataProvider, MeshElementRefinerParam>(
                    md, 1, mainMeshProvider, refinerParam),
                BlockTreeFieldGenerationPolicy::Propagate,
                cbField);
        if (param.outputBlockIndexInitializer.empty()) {
            REPORT_PROGRESS_STAGE("Interpolate dense field in all cubes");
            auto cubeIndex = 0u;
            bt.walkSubtrees(md.maxSubtreeDepth(), [&](const BlockId& subtreeRoot) {
                REPORT_PROGRESS_STAGES();
                if (!param.quiet) {
                    REPORT_PROGRESS_STAGE(string("Interpolate dense field in cube ") + boost::lexical_cast<string>(++cubeIndex));
                }
                DenseFieldInterpolator<N> dfi(btf);
                Vec2<real_type> fieldRange;
                vector<dfield_real> denseField;
                dfi.interpolate(fieldRange, denseField, subtreeRoot);
            });
        }
        else {
            REPORT_PROGRESS_STAGES();
            REPORT_PROGRESS_STAGE("Interpolate dense field in a cube");
            BlockIndex subtreeRootIndex;
            BOOST_ASSERT(param.outputBlockIndexInitializer.size() == N);
            boost::range::copy(param.outputBlockIndexInitializer, subtreeRootIndex.begin());
            bt.blockAt(subtreeRootIndex, param.outputBlockIndexLevel);
            auto subtreeRoot = bt.blockAt(subtreeRootIndex, param.outputBlockIndexLevel);
            DenseFieldInterpolator<N> dfi(btf);
            vector<dfield_real> denseField;
            Vec2<real_type> fieldRange;
            dfi.interpolate(fieldRange, denseField, subtreeRoot);

            REPORT_PROGRESS_STAGE("Write dense field to a tecplot file");
            auto subtreeNodes = md.blockTreeNodes(subtreeRoot); // TODO better: Copying!!!
            auto subtreePos = bt.blockPos(subtreeRoot);
            auto depth = subtreeNodes.maxDepth();
            using NodeCoord = typename BTN::NodeCoord;
            auto nodesPerEdge = IndexTransform<N>::template verticesPerEdge<NodeCoord>(depth);
            BOOST_ASSERT(denseField.size() == IndexTransform<N>::vertexCount(depth));
            NodeIndex nodeIndex;
            auto h = static_cast<real_type>(subtreePos.size()) / (1 << depth);
            Vec<N, real_type> subtreeOrigin;
            ScalarOrMultiIndex<N, real_type>::each_indexed(subtreePos.min(), [&](const real_type& x, unsigned int index) {
                subtreeOrigin[index] = x;
            });
            string fileName = param.denseFieldTecplotFileName(fieldName);
            ofstream os(fileName, ios::binary);
            if (os.fail())
                throw runtime_error(string("Failed to open output file '") + fileName + "'");
            cout << "Writing dense field to file '" << fileName << "'" << endl;
            os << "variables = " << (N == 3? "X Y Z": "X Y") << " " << fieldName << endl;
            os << "zone  i = " << nodesPerEdge << " j = " << nodesPerEdge;
            if (N == 3)
                os << " k = " << nodesPerEdge;
            os << endl
               << "T=\"  block1       \"" << endl;
            auto inode = 0u;
            do {
                auto vertexPos = subtreeOrigin + nodeIndex * h;
                for (auto i=0u; i<N; ++i)
                    os << vertexPos[i] << "\t";
                os << denseField[inode++] << endl;
            }
            while (incMultiIndex(nodeIndex, nodesPerEdge));
            BOOST_ASSERT(inode == denseField.size());
            os.close();

            REPORT_PROGRESS_STAGE("Write sparse field to a tecplot file");
            vector<real_type> sparseField;
            btf.fieldValues(fieldRange, sparseField, subtreeNodes);
            auto noFieldValue = btf.noFieldValue();
            boost::range::transform(sparseField, sparseField.begin(), [&](const auto& x) {
                return x == noFieldValue? NAN: x;
            });
            fileName = param.sparseFieldTecplotFileName(fieldName);
            os.open(fileName, ios::binary);
            if (os.fail())
                throw runtime_error(string("Failed to open output file '") + fileName + "'");
            cout << "Writing sparse field to file '" << fileName << "'" << endl;
            os << "variables = " << (N == 3? "X Y Z": "X Y") << " " << fieldName << endl;
            os << "zone  N = " << sparseField.size() << " E = 0 F=FEPOINT ET = " << (N==3? "BRICK": "QUAD") << endl
               << "T=\"  block1       \"" << endl;
            const auto& n2i = subtreeNodes.data().n2i;
            foreach_byindex32(i, sparseField) {
                auto vertexPos = subtreeOrigin + n2i[i] * h;
                for (auto i=0u; i<N; ++i)
                    os << vertexPos[i] << "\t";
                os << sparseField[i] << endl;
            }
        }
    }
    REPORT_PROGRESS_END();
}

} // anonymous namespace

int main(int argc, char *argv[])
{
    try {
        namespace po = boost::program_options;

        RunParameters runParam;

        auto po_value = [](auto& x) {
            return po::value(&x)->default_value(x);
        };

        auto spaceDimension = 3u;
        string outputIdString;

        po::options_description po_generic("Gerneric options");
        po_generic.add_options()
                ("help,h", "Produce help message")
                ("quiet,q", "Don't display progress of separate operations");
        po::options_description po_basic("Main options");
        po_basic.add_options()
                ("mesh_file", po::value(&runParam.meshFileName), "Mesh file name")
                ("depth", po_value(runParam.metadataBlockDepth), "Maximal depth within one block")
                ("refine", po_value(runParam.refinerParam), "Anisotropic element refinement (0=max, 1=none)")
                ("brefine", po_value(runParam.boundaryRefine), "Additional refinement at boundary (positive integer)")
                ("dim", po_value(spaceDimension), "Space dimension (1, 2, or 3)")
                ("out,o", po::value(&outputIdString), "Subtree id [field][:i,j,k,level] for which the dense field is to be output in the tecplot format")
                ("btec", po_value(runParam.saveBoundaryTecplot), "Save boundary to a tecplot file");

        po::positional_options_description po_pos;
        po_pos.add("mesh_file", 1);

        po::variables_map vm;
        auto po_alloptions = po::options_description().add(po_generic).add(po_basic);
        po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions)
                  .positional(po_pos).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << po_alloptions << "\n";
            return 0;
        }
        runParam.quiet = vm.count("quiet") > 0;

        if (runParam.meshFileName.empty())
            throw runtime_error("No mesh file name is specified");

        if (!outputIdString.empty()) {
            // Parse output subtree id
            regex rx("^([^:]*)?(:([^:]*))?$");
            smatch m;
            if (!regex_match(outputIdString, m, rx))
                throw runtime_error("Invalid output field and subtree id");
            BOOST_ASSERT(m.size() == 4);
            runParam.outputFieldName = m[1];
            string subtreeId = m[3];
            if (subtreeId.empty()) {
                runParam.outputBlockIndexInitializer.resize(spaceDimension);
                boost::range::fill(runParam.outputBlockIndexInitializer, 0);
                runParam.outputBlockIndexLevel = 0;
            }
            else {
                regex rxSubtree(spaceDimension == 3? "^(\\d+),(\\d+),(\\d+),(\\d+)$": "^(\\d+),(\\d+),(\\d+)$");
                if (!regex_match(subtreeId, m, rxSubtree))
                    throw runtime_error("Invalid output subtree block id");
                BOOST_ASSERT(m.size() == 2 + spaceDimension);
                runParam.outputBlockIndexInitializer.resize(spaceDimension);
                transform(m.begin()+1, m.end(), runParam.outputBlockIndexInitializer.begin(), [](const auto& x) {
                    return boost::lexical_cast<unsigned int>(x);
                });
                runParam.outputBlockIndexLevel = boost::lexical_cast<unsigned int>(m[spaceDimension+1]);
            }
        }

        REPORT_PROGRESS_STAGES();
        REPORT_PROGRESS_STAGE("Run the entire job");

        switch (spaceDimension) {
        case 1:
            run<1>(runParam);
            break;
        case 2:
            run<2>(runParam);
            break;
        case 3:
            run<3>(runParam);
            break;
        default:
            throw range_error("Invalid dimension");
        }
        return 0;
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}
