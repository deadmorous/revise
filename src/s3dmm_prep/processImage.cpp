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

#include "processImage.hpp"
#include "RunParameters.hpp"

#include <stdexcept>

#ifdef S3DMM_ENABLE_VTK

#include "s3dmm/MetadataProvider.hpp"
#include "s3dmm/BlockTreeFromImage.hpp"
#include "s3dmm/RectGridData.hpp"
#include "s3dmm/ImageStencil.hpp"
#include "s3dmm/ImageFunc.hpp"
#include "s3dmm/BlockTreeXWFieldFromImage.hpp"
#include "s3dmm/BlockTreeFieldFromFunc.hpp"
#include "s3dmm/BlockTreeFieldProvider.hpp"

#include <stdexcept>
#include <mutex>
#include <optional>

#include "filename_util.hpp"

#include "silver_bullets/templatize/resolve_template_args.hpp"

using namespace std;
using namespace s3dmm;

namespace {

template<unsigned int N>
class Sphere
{
public:
    using Index = MultiIndex<N, unsigned int>;
    Sphere(const Index& center, unsigned int radius) :
        m_center(center), m_radius(radius), r2(radius*radius)
    {}

    bool operator()(const Index& pos) const {
        auto d = pos.template convertTo<int>() - m_center.template convertTo<int>();
        return d*d <= r2;
    }

private:
    Index m_center;
    unsigned int m_radius;
    int r2;
};

template<unsigned int N>
BlockTreeFromImage<N, ImageStencil<N, Sphere<N>>> blockTreeFromImage(
    unsigned int domainSize, unsigned int minLevel, unsigned int maxLevel)
{
    auto imageSize = MultiIndex<N, unsigned int>::filled(domainSize);
    auto center = MultiIndex<N, unsigned int>::filled(domainSize>>1);
    auto d = domainSize / 20;
    auto radius = (domainSize >> 1) - d;
    Sphere<N> sphere(center, radius);
    using SphereImage = ImageStencil<N, Sphere<N>>;
    SphereImage img(imageSize, sphere);
    auto centerPos = ScalarOrMultiIndex<N, real_type>::fromMultiIndex(MultiIndex<N, real_type>());

    BoundingCube<N, real_type> pos(centerPos, 1);
    return BlockTreeFromImage<N, SphereImage>(imageSize, img, minLevel, maxLevel, pos);
}

class ColorToBool
{
public:
    explicit ColorToBool(int outsideDomainValue) : m_outsideDomainValue(outsideDomainValue)
    {}

    bool operator()(int x) const {
        return x != m_outsideDomainValue;
    }
private:
    int m_outsideDomainValue;
};

template<unsigned int N>
using ImageStencilFunc = ImageFunc<N, bool, int, ColorToBool>;

template<unsigned int N>
BlockTreeFromImage<N, ImageStencil<N, ImageStencilFunc<N>>> blockTreeFromImage(
    const RectGridData<N> *grid,
    const string& arrayName, int outsideDomainValue,
    unsigned int minLevel, unsigned int maxLevel)
{
    ImageStencilFunc<N> stencilFunc(grid, arrayName, ColorToBool(outsideDomainValue));
    using Stencil = ImageStencil<N, ImageStencilFunc<N>>;
    auto imageSize = grid->imageData().imageSize();
    Stencil stencil(imageSize, stencilFunc);
    return BlockTreeFromImage<N, Stencil>(imageSize, stencil, minLevel, maxLevel);
}

template <unsigned int N>
void processImageTemplate(const RunParameters& param)
{
    // constexpr auto domainSize = 100u;

    using BT = BlockTree<N>;
    using BTN = BlockTreeNodes<N, BT>;
    using BlockId = typename BT::BlockId;
    using NodeIndex = typename BTN::NodeIndex;

    auto [mainMeshFileName, hasTimeSteps] = firstOutputFrameFileName(param.meshFileName);
    auto outBaseName = s3dmmBaseName(param.meshFileName, param.outputDirectory);
    auto mainOutFileName = firstOutputFrameFileName(outBaseName).first;
    auto metadataFileName = mainOutFileName + ".s3dmm-meta";

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
    RectGridData<N> grid(mainMeshFileName);
    auto& imageData = grid.imageData();
    constexpr auto outsideDomainColor = -1;
    auto btg = blockTreeFromImage<N>(
        &grid, "Color", outsideDomainColor,
        param.metadataMaxFullLevel, param.metadataMaxLevel + param.metadataBlockDepth);
    using BTG = decltype (btg);
    MetadataProvider<N, BTG> mp(
        metadataFileName,
        param.metadataBlockDepth,
        param.metadataMaxLevel,
        param.metadataMaxFullLevel,
        btg);
    if (!param.quiet)
        mp.setSubtreeNodesProgressCallback(cbSubtreeNodes);
    auto& md = mp.metadata();

    // Generate boundary shape field
    REPORT_PROGRESS_STAGE("Generate boundary shape field");
    shared_ptr<typename BlockTreeFieldProvider<N>::Timers> fieldGenTimers;
    {
        auto shapeFunc = [](int x) { return make_real(x == outsideDomainColor? -1: 1); };
        using Func = ImageFunc<N, real_type, int, decltype(shapeFunc)>;
        BlockTreeFieldProvider<N> shapeFieldProvider(
            md,
            mainOutFileName + ".s3dmm-field#shape",
            BlockTreeXWFieldFromImage<N, Func>(
                md,
                Func(&grid, "Color", shapeFunc),
                imageData.imageSize()),
            BlockTreeFieldGenerationPolicy::Propagate,
            cbField, fieldGenTimers);
    }

    // Generate dataset fields
    REPORT_PROGRESS_STAGE("Generate dataset fields");
    vector<unsigned int> fieldIds;
    auto arrayCount = imageData.arrayCount();
    auto toReal = [](double x) { return make_real(x); };
    for (auto arrayId=0u; arrayId<arrayCount; ++arrayId) {
        if (imageData.arrayInfo(arrayId).type == type_ordinal_v<double>) {
            auto fieldName = imageData.arrayName(arrayId);
            fieldIds.push_back(arrayId);
            using Func = ImageFunc<N, real_type, double, decltype(toReal)>;
            BlockTreeFieldProvider<N> shapeFieldProvider(
                md,
                mainOutFileName + ".s3dmm-field#" + fieldName,
                BlockTreeXWFieldFromImage<N, Func>(
                    md,
                    Func(&grid, fieldName, toReal),
                    imageData.imageSize()),
                BlockTreeFieldGenerationPolicy::Propagate,
                cbField, fieldGenTimers);
        }
    }

    // Generate auxiliary fields
    REPORT_PROGRESS_STAGE("Generate auxiliary fields");
    string auxCoordFieldNames[] = { "coord_x", "coord_y", "coord_z" };
    BOOST_STATIC_ASSERT(N <= 3);
    for (auto d=0u; d<N; ++d) {
        auto fieldName = auxCoordFieldNames[d];
        ImageToBlockIndexTransform<N> i2b(
            imageData.imageSize(),
            BlockId(),
            param.metadataBlockDepth,
            md.blockTree().blockPos(BlockId()));
        using vector_type = ScalarOrMultiIndex_t<N, real_type>;
        auto func = [&i2b, d](const vector_type& pos) {
            return ScalarOrMultiIndex<N, real_type>::element(pos, d);
        };
        using Func = decltype (func);
        BlockTreeFieldProvider<N> shapeFieldProvider(
            md,
            mainOutFileName + ".s3dmm-field#" + fieldName,
            BlockTreeFieldFromFunc<N, Func>(
                md,
                func),
            BlockTreeFieldGenerationPolicy::Propagate,
            cbField, fieldGenTimers);
    }

    REPORT_PROGRESS_STAGE("Compute total field ranges");
    vector<optional<Vec2<real_type>>> fieldRanges(fieldIds.size());
    auto frame = 1u;    // TODO: time steps
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

        foreach_byindex32(i, fieldIds) {
            auto fieldId = fieldIds[i];
            auto fieldName = imageData.arrayName(fieldId);
            for (auto timeFrame=0u; timeFrame<frame; ++timeFrame) {
                auto fieldFileName = frameOutputFileName(outBaseName, timeFrame, hasTimeSteps) + ".s3dmm-field#" + fieldName;
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
            auto s = splitFileName(outBaseName);
            infoFileName = path(get<0>(s)).append(get<1>(s)).append(get<1>(s) + get<2>(s) + ".s3dmm-fields");
        }
        else
            infoFileName = outBaseName + ".s3dmm-fields";
        ofstream os(infoFileName);
        if (os.fail())
            throw runtime_error(string("Failed to open output file '") + infoFileName + "'");
        os << "time_steps\n"
           << frame << endl << endl;
        os << "field\tmin\tmax\tanimated" << endl;
        foreach_byindex32(i, fieldIds) {
            auto fieldId = fieldIds[i];
            os << imageData.arrayName(fieldId) << '\t';
            auto fieldRange = fieldRanges[i];
            if (fieldRange) {
                auto r = fieldRange.value();
                os << r[0] << '\t' << r[1];
            }
            else
                os << "-\t-";
            os << '\t' << (hasTimeSteps? 1: 0) << endl;
        }
        os << "shape\t-1\t1\t0" << endl;
        auto rootPos = md.blockTree().blockPos(BlockId());
        for (auto d=0u; d<N; ++d) {
            auto xmin = ScalarOrMultiIndex<N, real_type>::element(rootPos.min(), d);
            auto xmax = ScalarOrMultiIndex<N, real_type>::element(rootPos.max(), d);
            os << auxCoordFieldNames[d] << '\t' << xmin << '\t' << xmax << "\t0" << endl;
        }
    }
    REPORT_PROGRESS_END();
}

struct callProcessImageTemplate {
    template<unsigned int N> void operator()(const RunParameters& param) const {
        processImageTemplate<N>(param);
    }
};

} // anonymous namespace

void processImage(const RunParameters& param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Run the entire job");
    silver_bullets::resolve_template_args<
        integer_sequence<unsigned int, 1,2,3>>(
        make_tuple(param.spaceDimension), callProcessImageTemplate(), param);
}
#else // S3DMM_ENABLE_VTK

void processImage(const RunParameters&)
{
    throw std::runtime_error("processImage() failed: please specify the S3DMM_ENABLE_VTK option in CMake");
}
#endif // S3DMM_ENABLE_VTK
