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

#pragma once

#include "BlockTree.hpp"
#include "ProgressReport.hpp"
#include "MeshBoundaryWalker.hpp"
#include "FullBlockTreeGenerator.hpp"
#include "elementScalarSize.hpp"

#include "silver_bullets/fs_ns_workaround.hpp"

#include <fstream>


namespace s3dmm {

template <unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeFromMesh
{
public:
    struct BlockTreeProgressCallbackData
    {
        unsigned int zone;
        unsigned int zoneCount;
        unsigned int zoneElement;
        unsigned int zoneElementCount;
        unsigned int blockTreeSize;
    };
    BlockTreeFromMesh(
            const std::string& metadataFileName,
            MeshProvider& meshProvider,
            unsigned int minTreeDepth,
            unsigned int maxTreeDepth,
            const MeshRefinerParam& meshRefinerParam,
            std::function<const MeshBoundaryExtractor&()> mbxGetter,
            unsigned int boundaryRefine)
        :
        m_metadataFileName(metadataFileName),
        m_meshProvider(meshProvider),
        m_minTreeDepth(minTreeDepth),
        m_maxTreeDepth(maxTreeDepth),
        m_meshRefinerParam(meshRefinerParam),
        m_mbxGetter(mbxGetter),
        m_boundaryRefine(boundaryRefine)
    {}

    void setBlockTreeProgressCallback(
        const std::function<void(const BlockTreeProgressCallbackData&)>& blockTreeProgressCallback)
    {
        m_blockTreeProgressCallback = blockTreeProgressCallback;
    }

    MeshProvider& meshProvider() const {
        return m_meshProvider;
    }

    BlockTree<N> makeBlockTree() const
    {
        REPORT_PROGRESS_STAGES();
        REPORT_PROGRESS_STAGE("Open mesh");
        auto cache = m_meshProvider.makeCache();
        auto cvars = m_meshProvider.coordinateVariables();
        if (cvars.size() != N)
            throw std::runtime_error("Invalid dimension in the mesh file");
        auto bb = maybeReadBoundingBox();

        auto& meshBoundaryExtractor = m_mbxGetter();
        if (bb.empty()) {
            REPORT_PROGRESS_STAGE("Compute bounding box");
            for (auto zone : m_meshProvider.zones(cache, cvars)) {
                for (auto nodePos : zone.nodes()) {
                    vector_type r;
                    BOOST_ASSERT(nodePos.size() == N);
                    boost::range::copy(nodePos, &ScalarOrMultiIndex<N, real_type>::element(r, 0));
                    bb << r;
                }
            }

            writeBoundingBox(bb, "-orig");

            // Adjust bounding box by adding small space around the domain,
            // required for surrounding the boundary with small cubes
            for (auto& zone : meshBoundaryExtractor.zones()) {
                switch (zone.elementType()) {
                    case MeshElementType::Hexahedron:
                        adjustBoundingBox(bb, zone.template data<MeshElementType::Hexahedron>());
                        break;
                    case MeshElementType::Quad:
                        adjustBoundingBox(bb, zone.template data<MeshElementType::Quad>());
                        break;
                    default:
                        BOOST_ASSERT(false);
                        throw std::runtime_error("BlockTreeFromMesh::makeBlockTree(): Unsupported element type");
                }
            }

            writeBoundingBox(bb);
        }

        REPORT_PROGRESS_STAGE("Generate block tree from mesh");
        auto result = FullBlockTreeGenerator<N>(m_minTreeDepth, bb).makeBlockTree();
        std::vector<real_type> nodes;
        auto izone = 0u;
        auto zoneCount = m_meshProvider.zoneCount();
        std::vector<real_type> interpolatedRefined;
        for (auto zone : m_meshProvider.zones(cache, cvars)) {
            auto elements = zone.elements();
            auto elementCount = zone.elementCount();
            nodes.resize(zone.nodesPerElement() * N);
            auto izoneElement = 0u;

            auto processOriginalZoneElements = [&]() {
                for (auto it=elements.begin(); it!=elements.end(); ++it) {
                    auto elementNodes = it.elementNodes();
                    auto elementSize = elementScalarSize<N>(elementNodes);
                    auto level = limitLevel(result.levelForSize(elementSize));
                    for (auto nodePos : elementNodes)
                        result.ensureBlockAt(*reinterpret_cast<const vector_type*>(nodePos.begin()), level);
                }
            };

            auto processRefinedZoneElements = [&](const auto& meshRefiner) {
                for (auto it=elements.begin(); it!=elements.end(); ++it) {
                    auto elementNodes = it.elementNodes();
                    auto nd = nodes.data();
                    for (auto nodePos : elementNodes) {
                        BOOST_ASSERT(nodePos.size() == N);
                        boost::range::copy(nodePos, nd);
                        nd += N;
                    }
                    auto refined = meshRefiner.refine(nodes.data(), N);
                    auto refinedNodeCount=refined.count();
                    auto interpolatedRefinedSize = refinedNodeCount * N;
                    if (interpolatedRefined.size() < interpolatedRefinedSize)
                        interpolatedRefined.resize(interpolatedRefinedSize);
                    refined.interpolate(interpolatedRefined.data(), N, nodes.data(), N);
                    auto interpolatedRefinedPtr = interpolatedRefined.data();
                    auto scalarSize = refined.size();
                    if (scalarSize <= 0)
                        throw std::runtime_error("Cannot estimate level because bounding box size is zero");
                    auto level = limitLevel(result.levelForSize(scalarSize));
                    for (auto iRefinedNode=0u, refinedNodeCount=refined.count(); iRefinedNode<refinedNodeCount; ++iRefinedNode, interpolatedRefinedPtr+=N)
                        result.ensureBlockAt(*reinterpret_cast<vector_type*>(interpolatedRefinedPtr), level);
                    if (m_blockTreeProgressCallback)
                        m_blockTreeProgressCallback({
                            izone,
                            zoneCount,
                            izoneElement,
                            elementCount,
                            static_cast<unsigned int>(result.data().children.size())
                        });
                    ++izoneElement;
                }
            };

            if (m_meshRefinerParam.needRefiner()) {
                switch (zone.elementType()) {
                    case MeshElementType::Triangle:
                        processRefinedZoneElements(m_meshRefinerParam.template refiner<MeshElementType::Triangle>());
                        break;
                        throw std::runtime_error("No mesh element refiner exists for triangles");
                    case MeshElementType::Quad:
                        processRefinedZoneElements(m_meshRefinerParam.template refiner<MeshElementType::Quad>());
                        break;
                    case MeshElementType::Tetrahedron:
                        processRefinedZoneElements(m_meshRefinerParam.template refiner<MeshElementType::Tetrahedron>());
                        break;
                        throw std::runtime_error("No mesh element refiner exists for tetrahedra");
                    case MeshElementType::Hexahedron:
                        processRefinedZoneElements(m_meshRefinerParam.template refiner<MeshElementType::Hexahedron>());
                        break;
                }
            }
            else
                processOriginalZoneElements();

            ++izone;
        }

        // Refine block tree at the boundary
        for (auto& zone : meshBoundaryExtractor.zones()) {
            switch (zone.elementType()) {
                case MeshElementType::Hexahedron:
                    refineBlockTree(result, zone.template data<MeshElementType::Hexahedron>());
                    break;
                case MeshElementType::Quad:
                    refineBlockTree(result, zone.template data<MeshElementType::Quad>());
                    break;
                default:
                    BOOST_ASSERT(false);
                    throw std::runtime_error("BlockTreeFromMesh::makeBlockTree(): Unsupported element type");
            }
        }

        return result;
    }

    unsigned int maxTreeDepth() const {
        return m_maxTreeDepth;
    }

private:
    using vector_type = typename BoundingBox<N, real_type>::vector_type;

    std::string m_metadataFileName;
    MeshProvider& m_meshProvider;
    unsigned int m_minTreeDepth;
    unsigned int m_maxTreeDepth;
    MeshRefinerParam m_meshRefinerParam;
    std::function<const MeshBoundaryExtractor&()> m_mbxGetter;
    unsigned int m_boundaryRefine;

    std::function<void(const BlockTreeProgressCallbackData&)> m_blockTreeProgressCallback;

    unsigned int limitLevel(unsigned int level) const {
        return level > m_maxTreeDepth? m_maxTreeDepth: level;
    };

    template<MeshElementType elementType>
    void adjustBoundingBox(
            BoundingBox<N, real_type>&,
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>&,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension != N, int> = 0) const
    {
        throw std::runtime_error("Invalid space dimension in the zone of the input file");
    }

    template<MeshElementType elementType>
    void adjustBoundingBox(
            BoundingBox<N, real_type>& bb,
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>& zd,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension == N, int> = 0) const
    {
        MeshBoundaryWalker bwalker(m_boundaryRefine);
        bwalker.walkRefinedNodes(
                    zd,
                    [&](const vector_type& pos, const Vec<N, real_type>& n, real_type size)
        {
            auto x = pos;
            reinterpret_cast<Vec<N, real_type>&>(x) += n*size;
            bb << x;
        });
    }

    template<MeshElementType elementType>
    void refineBlockTree(
            BlockTree<N>&,
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>&,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension != N, int> = 0) const
    {
        throw std::runtime_error("Invalid space dimension in the zone of the input file");
    }

    template<MeshElementType elementType>
    void refineBlockTree(
            BlockTree<N>& bt,
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>& zd,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension == N, int> = 0) const
    {
        MeshBoundaryWalker bwalker(m_boundaryRefine);
        bwalker.walkRefinedNodes(
                    zd,
                    [&](const vector_type& pos, const Vec<N, real_type>& n, real_type size)
        {
            auto level = limitLevel(bt.levelForSize(size));
            if (level > 0)
                --level;
            bt.ensureBlockAt(pos, level);
            auto addPoint = [&](const Vec<N, real_type>& d) {
                auto x = pos;
                reinterpret_cast<Vec<N, real_type>&>(x) += d;
                bt.ensureBlockAt(x, level);
            };
            auto d = n*(make_real(0.5)*size);
            addPoint(d);
            addPoint(-d);
        });
    }

    std::string boundingBoxFileName(const std::string& additionalSuffix = std::string()) const {
        return m_metadataFileName + "-bb" + additionalSuffix;
    }

    BoundingBox<N, real_type> maybeReadBoundingBox() const
    {
        using namespace std::experimental::filesystem;
        BoundingBox<N, real_type> bb;
        auto bbFileName = boundingBoxFileName();
        if (exists(bbFileName)) {
            std::ifstream is(bbFileName);
            is.exceptions(std::istream::failbit);
            for (auto i=0u; i<2; ++i) {
                vector_type x;
                for (auto d=0u; d<N; ++d)
                    is >> ScalarOrMultiIndex<N, real_type>::element(x, d);
                bb << x;
            }
        }
        return bb;
    }

    void writeBoundingBox(const BoundingBox<N, real_type>& bb, const std::string& additionalSuffix = std::string()) const
    {
        auto bbFileName = boundingBoxFileName(additionalSuffix);
        std::ofstream os(bbFileName);
        if (os.fail())
            throw std::runtime_error(std::string("Failed to open bounding box file '") + bbFileName + "' for output");
        vector_type x[2] = { bb.min(), bb.max() };
        for (auto i=0u; i<2; ++i) {
            for (auto d=0u; d<N; ++d) {
                if (d > 0)
                    os << '\t';
                os << ScalarOrMultiIndex<N, real_type>::element(x[i], d);
            }
            os << std::endl;
        }
    }
};

} // s3dmm
