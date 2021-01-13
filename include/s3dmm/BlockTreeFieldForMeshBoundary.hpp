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

#include "MeshBoundaryWalker.hpp"
#include "Metadata.hpp"
#include "faceNormalVector.hpp"

#include <experimental/filesystem>
#include <fstream>
#include <memory>

namespace s3dmm {

template<unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeFieldForMeshBoundary : boost::noncopyable
{
public:
    using BlockId = typename BlockTree<N>::BlockId;
    using BT = typename Metadata<N>::BT;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeFieldForMeshBoundary(
            const Metadata<N>& metadata,
            std::function<const MeshBoundaryExtractor&()> mbxGetter,
            unsigned int boundaryRefine) :
        m_metadata(metadata),
        m_mbxGetter(mbxGetter),
        m_boundaryRefine(boundaryRefine)
    {
    }

    void generate(
            real_type *fieldValues,
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            real_type noFieldValue,
            unsigned int /*processedChildSubtrees*/) const
    {
#if 0
        // Fill all values with 2 (maximum positive distance to the boundary)
        std::fill(fieldValues, fieldValues+nodeCount, make_real(2));
        auto tol = make_real(1e-6)*subtreePos.size();
        auto maxLevel = subtreeRoot.level + subtreeNodes.maxDepth();
        auto& blockTree = m_metadata.blockTree();

        // Fill all values in blocks around nodes with -2 (maximum negative distance to the boundary)
        for (auto zone : m_meshProvider.zones(cache(), vars())) {
            auto processOriginalZoneNodes = [&]() {
                for (auto& node : zone.nodes()) {
                    auto& pos = *reinterpret_cast<const vector_type*>(node.begin());
                    if (subtreePos.contains(pos, tol)) {
                        auto blockId = blockTree.depthLimitedBlockAt(pos, maxLevel);
                        if (!isSameOrChildOf(blockId, subtreeRoot))
                            continue;
                        subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                    unsigned int /*localNodeNumber*/,
                                                    const NodeIndex& /*nodeIndex*/,
                                                    std::size_t nodeNumber)
                        {
                            fieldValues[nodeNumber] = make_real(-2);
                        });
                    }
                }
            };
            auto processRefinedZoneElements = [&](const auto& meshRefiner) {
                auto elements = zone.elements();
                m_nodes.resize(zone.nodesPerElement() * N);
                for (auto it=elements.begin(); it!=elements.end(); ++it) {
                    auto elementNodes = it.elementNodes();
                    auto nd = m_nodes.data();
                    for (auto nodePos : elementNodes) {
                        BOOST_ASSERT(nodePos.size() == N);
                        boost::range::copy(nodePos, nd);
                        nd += N;
                    }
                    auto refined = meshRefiner.refine(m_nodes.data(), N);
                    auto refinedNodeCount=refined.count();
                    auto interpolatedRefinedSize = refinedNodeCount * N;
                    if (m_interpolatedRefined.size() < interpolatedRefinedSize)
                        m_interpolatedRefined.resize(interpolatedRefinedSize);
                    refined.interpolate(m_interpolatedRefined.data(), N, m_nodes.data(), N);
                    auto interpolatedRefinedPtr = m_interpolatedRefined.data();
                    for (auto iRefinedNode=0u; iRefinedNode<refinedNodeCount; ++iRefinedNode, interpolatedRefinedPtr+=N) {
                        auto& pos = *reinterpret_cast<const vector_type*>(interpolatedRefinedPtr);
                        if (subtreePos.contains(pos, tol)) {
                            auto blockId = blockTree.depthLimitedBlockAt(pos, maxLevel);
                            if (!isSameOrChildOf(blockId, subtreeRoot))
                                continue;
                            auto blockPos = blockTree.blockPos(blockId);
                            subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                        unsigned int /*localNodeNumber*/,
                                                        const NodeIndex& /*nodeIndex*/,
                                                        std::size_t nodeNumber) {
                                fieldValues[nodeNumber] = make_real(-2);
                            });
                        }
                    }
                }
            };
            auto meshRefiner = m_meshRefinerParam.template refiner<MeshElementType::Hexahedron/*zone.elementType()*/>();
            processRefinedZoneElements(meshRefiner);
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
                processOriginalZoneNodes();
        }
#endif // 0

        // Fill all values in blocks around boundary nodes with distances from the nodes to the boundary
        for (auto& zone : m_mbxGetter().zones()) {
            switch (zone.elementType()) {
            case MeshElementType::Hexahedron: {
                processZone(zone.template data<MeshElementType::Hexahedron>(),
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes);
                break;
            }
            case MeshElementType::Quad: {
                processZone(zone.template data<MeshElementType::Quad>(),
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes);
                break;
            }
            default:
                BOOST_ASSERT(false);
                throw std::runtime_error("BlockTreeFieldForMeshBoundary::generate(): Unsupported element type");
            }
        }

        // Average computed distances
        auto nodeCount = subtreeNodes.nodeCount();
        std::transform(
                    fieldValues, fieldValues+nodeCount,
                    weightValues,
                    fieldValues,
                    [noFieldValue](real_type field, real_type weight)
        {
            return weight > 0? field / weight: noFieldValue;
        });

        // Normalize distances
        auto minDist = make_real(0);
        auto maxDist = make_real(0);
        for (auto i=0u; i<nodeCount; ++i) {
            if (weightValues[i] > 0) {
                auto& dist = fieldValues[i];
                if (minDist > dist)
                    minDist = dist;
                else if (maxDist < dist)
                    maxDist = dist;
            }
        }
        if (minDist < make_real(0) && maxDist > make_real(0)) {
            for (auto i=0u; i<nodeCount; ++i) {
                auto& weight = weightValues[i];
                if (weight > 0) {
                    auto& dist = fieldValues[i];
                    if (dist > 0)
                        dist /= maxDist;
                    else
                        dist /= -minDist;
                    weight = make_real(1);
                }
            }
        }
    }

private:
    const Metadata<N>& m_metadata;
    std::function<const MeshBoundaryExtractor&()> m_mbxGetter;
    unsigned int m_boundaryRefine;

    template<MeshElementType elementType>
    void processZone(
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>&,
            real_type*,
            real_type*,
            const BlockId&,
            const BoundingCube<N, real_type>&,
            const BlockTreeNodes<N, BT>&,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension != N, int> = 0) const
    {
        throw std::runtime_error("Invalid space dimension in the zone of the input file");
    }

    template<MeshElementType elementType>
    void processZone(
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>& zd,
            real_type *fieldValues,
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            std::enable_if_t<MeshElementTraits<elementType>::SpaceDimension == N, int> = 0) const
    {
        using vector_type = typename BlockTree<N>::vector_type;
        auto tol = make_real(1e-6)*subtreePos.size();
        auto maxLevel = subtreeRoot.level + subtreeNodes.maxDepth();
        auto& blockTree = m_metadata.blockTree();

        MeshBoundaryWalker(m_boundaryRefine).walkRefinedNodes(
                    zd,
                    [&](const vector_type& pos, const Vec<N, real_type>& n, real_type size)
        {
            if (norm2(n) <= 0)
                // Skip degenerate face
                return;
            auto d = size*n;
            auto r = pos-d;
            BlockId passedBlocks[3];
            for (auto i=0; i<3; ++i, r+=d) {
                auto isBlockPassed = [&](const BlockId& blockId) {
                    for (auto j=0; j<i; ++j)
                        if (passedBlocks[j] == blockId)
                            return true;
                    return false;
                };
                if (subtreePos.contains(r, tol)) {
                    auto blockId = blockTree.depthLimitedBlockAt(r, maxLevel);
                    passedBlocks[i] = blockId;
                    if (isBlockPassed(blockId))
                        continue;
                    if (!isSameOrChildOf(blockId, subtreeRoot))
                        continue;
                    auto blockPos = blockTree.blockPos(blockId);
                    subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                unsigned int /*localNodeNumber*/,
                                                const NodeIndex& nodeIndex,
                                                std::size_t nodeNumber)
                    {
                        // Compute block node position
                        auto blockNodePos = blockTree.vertexPos(nodeIndex.template convertTo<unsigned int>(), maxLevel);
                        fieldValues[nodeNumber] += n * (blockNodePos - pos);
                        ++weightValues[nodeNumber];
                    });
                }
            }
        });
    }

    mutable std::vector<real_type> m_nodes;
    mutable std::vector<real_type> m_interpolatedRefined;

    static bool isSameOrChildOf(const BlockId& maybeChild, const BlockId& maybeParent)
    {
        if (maybeChild.level < maybeParent.level)
            return false;
        auto loc = maybeChild.location >> (maybeChild.level - maybeParent.level);
        return loc == maybeParent.location;
    }

};

} // s3dmm
