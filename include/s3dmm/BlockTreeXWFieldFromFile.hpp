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

#include "Metadata.hpp"
#include "MeshElementType.hpp"

#include <boost/range/algorithm/copy.hpp>

namespace s3dmm {

// Note: XWField means "extrapolated weighed field",
// according to the algorithm used to compute field values at block tree nodes.
template<unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeXWFieldFromFile : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeXWFieldFromFile(
            const Metadata<N>& metadata,
            unsigned int fieldIndex,
            MeshProvider& meshProvider,
            const MeshRefinerParam& meshRefinerParam) :
        m_metadata(metadata),
        m_fieldIndex(fieldIndex),
        m_meshProvider(meshProvider),
        m_meshRefinerParam(meshRefinerParam)
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
        const auto Tol = make_real(1e-5);
        const auto _2Tol = make_real(2)*Tol;
        auto tol = make_real(1e-6)*subtreePos.size();
        auto nodeCount = subtreeNodes.nodeCount();
        auto maxLevel = subtreeRoot.level + subtreeNodes.maxDepth();
        auto& blockTree = m_metadata.blockTree();
        for (auto zone : m_meshProvider.zones(cache(), vars())) {
            auto processOriginalZoneNodes = [&]() {
                for (auto& node : zone.nodes()) {
                    auto& pos = *reinterpret_cast<const vector_type*>(node.begin());
                    if (subtreePos.contains(pos, tol)) {
                        auto& v = node[N];
                        auto blockId = blockTree.depthLimitedBlockAt(pos, maxLevel);
                        if (!isSameOrChildOf(blockId, subtreeRoot))
                            continue;
                        auto blockPos = blockTree.blockPos(blockId);
                        vector_type param = (pos - blockPos.min()) / blockPos.size();
                        subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                    unsigned int localNodeNumber,
                                                    const NodeIndex& nodeIndex,
                                                    std::size_t nodeNumber) {
                            auto weight = make_real(1);
                            for (auto d=N-1; d!=~0u; --d, localNodeNumber>>=1) {
                                auto t = ScalarOrMultiIndex<N, real_type>::element(param, d);
                                if ((localNodeNumber & 1) == 0u)
                                    t = 1-t;
                                weight *= t;
                            }
                            // Note: Do not allow weight less than _2Tol in a cube,
                            // because that would lead to possible noFieldValue in the cube,
                            // which in turn will be interpreted as a hole in the dense field.
                            if (weight < _2Tol)
                                weight = _2Tol;
                            fieldValues[nodeNumber] += v*weight;
                            weightValues[nodeNumber] += weight;
                        });
                    }
                }
            };
            auto processRefinedZoneElements = [&](const auto& meshRefiner) {
                auto elements = zone.elements();
                m_nodes.resize(zone.nodesPerElement() * (N+1));
                for (auto it=elements.begin(); it!=elements.end(); ++it) {
                    auto elementNodes = it.elementNodes();
                    auto nd = m_nodes.data();
                    for (auto nodePos : elementNodes) {
                        BOOST_ASSERT(nodePos.size() == N + 1);
                        boost::range::copy(nodePos, nd);
                        nd += N + 1;
                    }
                    auto refined = meshRefiner.refine(m_nodes.data(), N+1);
                    auto refinedNodeCount=refined.count();
                    auto interpolatedRefinedSize = refinedNodeCount * (N+1);
                    if (m_interpolatedRefined.size() < interpolatedRefinedSize)
                        m_interpolatedRefined.resize(interpolatedRefinedSize);
                    refined.interpolate(m_interpolatedRefined.data(), N+1, m_nodes.data(), N+1);
                    auto interpolatedRefinedPtr = m_interpolatedRefined.data();
                    for (auto iRefinedNode=0u; iRefinedNode<refinedNodeCount; ++iRefinedNode, interpolatedRefinedPtr+=N+1) {
                        auto& pos = *reinterpret_cast<const vector_type*>(interpolatedRefinedPtr);
                        if (subtreePos.contains(pos, tol)) {
                            auto& v = interpolatedRefinedPtr[N];
                            auto blockId = blockTree.depthLimitedBlockAt(pos, maxLevel);
                            if (!isSameOrChildOf(blockId, subtreeRoot))
                                continue;
                            auto blockPos = blockTree.blockPos(blockId);
                            vector_type param = (pos - blockPos.min()) / blockPos.size();
                            subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                        unsigned int localNodeNumber,
                                                        const NodeIndex& nodeIndex,
                                                        std::size_t nodeNumber) {
                                auto weight = make_real(1);
                                for (auto d=N-1; d!=~0u; --d, localNodeNumber>>=1) {
                                    auto t = ScalarOrMultiIndex<N, real_type>::element(param, d);
                                    if ((localNodeNumber & 1) == 0u)
                                        t = 1-t;
                                    weight *= t;
                                }
                                // Note: Do not allow weight less than _2Tol in a cube,
                                // because that would lead to possible noFieldValue in the cube,
                                // which in turn will be interpreted as a hole in the dense field.
                                if (weight < _2Tol)
                                    weight = _2Tol;
                                fieldValues[nodeNumber] += v*weight;
                                weightValues[nodeNumber] += weight;
                            });
                        }
                    }
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
                processOriginalZoneNodes();
        }

        std::transform(
                    fieldValues, fieldValues+nodeCount,
                    weightValues,
                    fieldValues,
                    [Tol, noFieldValue](real_type field, real_type weight)
        {
            return weight < Tol? noFieldValue: field / weight;
        });
    }

private:
    const Metadata<N>& m_metadata;
    unsigned int m_fieldIndex;
    MeshProvider& m_meshProvider;
    const MeshRefinerParam& m_meshRefinerParam;

    mutable typename MeshProvider::Cache m_cache;
    typename MeshProvider::Cache cache() const
    {
        if (!m_cache)
            m_cache = m_meshProvider.makeCache();
        return m_cache;
    }

    mutable std::vector<unsigned int> m_vars;
    const std::vector<unsigned int>& vars() const
    {
        if (m_vars.empty()) {
            m_vars = m_meshProvider.coordinateVariables();
            if (m_vars.size() != N)
                throw std::runtime_error("Invalid dimension in the mesh file");
            m_vars.push_back(m_fieldIndex);
        }
        return m_vars;
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
