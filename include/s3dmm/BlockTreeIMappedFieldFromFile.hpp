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
#include "elementBoundingBox.hpp"
#include "ElementApprox.hpp"
#include "binary_io.hpp"

#include <boost/range/algorithm/copy.hpp>

namespace s3dmm {

// Note: IMappedField means "interpolated mapped field",
// according to the algorithm used to compute field values at block tree nodes.
template<unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeIMappedFieldFromFile : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeIMappedFieldFromFile(
            const Metadata<N>& metadata,
            unsigned int fieldIndex,
            MeshProvider& meshProvider) :
        m_metadata(metadata),
        m_fieldIndex(fieldIndex),
        m_meshProvider(meshProvider)
    {
    }

private:
    template<MeshElementType elementType>
    void mapZone(
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            const typename MeshProvider::Zone& zone,
            const std::vector<BoundingCube<N, real_type>>& processedAreas,
            std::ostream& fieldMap) const
    {
        BinaryWriter writer(fieldMap);
        auto maxLevel = subtreeRoot.level + subtreeNodes.maxDepth();
        auto& blockTree = m_metadata.blockTree();
        BoundingBox<N, real_type> subtreeBox;
        subtreeBox << subtreePos.min() << subtreePos.max();

        auto elements = zone.elements();

        auto gridSize = subtreePos.size() / (1 << subtreeNodes.maxDepth());
        const BlockIndex ione = BlockIndex::filled(1u);

        auto ielement = 0u;
        for (auto it=elements.begin(); it!=elements.end(); ++it, ++ielement) {
            auto elementNodes = it.elementNodes();

            auto elementBox = elementBoundingBox<N>(elementNodes) & subtreeBox;

            if (!elementBox.empty() &&
                    !std::any_of(processedAreas.begin(), processedAreas.end(), [&](const auto& bc) {
                        return bc.contains(elementBox);
                    }))
            {
                auto minPos = ((elementBox.min()-subtreePos.min())/gridSize);
                auto maxPos = ((elementBox.max()-subtreePos.min())/gridSize);
                auto imin = ScalarOrMultiIndex<N, real_type>::toMultiIndex(minPos).template convertTo<unsigned int>();
                auto imax = ScalarOrMultiIndex<N, real_type>::toMultiIndex(maxPos).template convertTo<unsigned int>() + ione;

                auto elementApprox = makeElementApprox<N, elementType>(elementNodes);

                subtreeNodes.walkIndexBoxNodes(imin, imax, [&](const NodeIndex& nodeIndex, std::size_t nodeNumber) {
                    if (weightValues[nodeNumber] > 0)
                        return;
                    auto nodePos = reinterpret_cast<const Vec<N,real_type>&>(subtreePos.min()) + nodeIndex*gridSize;
                    Vec<N,real_type> param;
                    if (elementApprox.param(param, nodePos))
                        writer << ielement << static_cast<unsigned int>(nodeNumber) << param;
                });
            }
        }
        writer << ~0u;
    }

    template<MeshElementType elementType>
    void generateZone(
            real_type *fieldValues,
            real_type *weightValues,
            const typename MeshProvider::Zone& zone,
            std::istream& fieldMap) const
    {
        BinaryReader reader(fieldMap);
        auto elements = zone.elements();
        while (true) {
            auto ielement = reader.read<unsigned int>();
            if (ielement == ~0u)
                break;
            auto nodeNumber = reader.read<unsigned int>();
            auto param = reader.read<Vec<N,real_type>>();
            auto elementIter = elements.begin() + ielement;
            auto elementNodes = elementIter.elementNodes();
            ElementApprox<N, elementType>::approx(fieldValues+nodeNumber, elementNodes, param);
            weightValues[nodeNumber] = make_real(1);
        }
    }

public:

    void generate(
            real_type *fieldValues,
            real_type *weightValues,
            unsigned int subtreeNodeCount,
            real_type noFieldValue,
            std::istream& fieldMap) const
    {
        std::vector<unsigned int> vars = { m_fieldIndex };
        for (auto zone : m_meshProvider.zones(cache(), vars)) {
            switch (zone.elementType()) {
            case MeshElementType::Triangle:
                generateZone<MeshElementType::Triangle>(
                            fieldValues, weightValues, zone, fieldMap);
                break;
            case MeshElementType::Quad:
                generateZone<MeshElementType::Quad>(
                            fieldValues, weightValues, zone, fieldMap);
                break;
            case MeshElementType::Tetrahedron:
                generateZone<MeshElementType::Tetrahedron>(
                            fieldValues, weightValues, zone, fieldMap);
                break;
            case MeshElementType::Hexahedron:
                generateZone<MeshElementType::Hexahedron>(
                            fieldValues, weightValues, zone, fieldMap);
                break;
            }
        }

        const auto Tol = make_real(1e-5);
        std::transform(
                    fieldValues, fieldValues+subtreeNodeCount,
                    weightValues,
                    fieldValues,
                    [Tol, noFieldValue](real_type field, real_type weight)
        {
            return weight < Tol? noFieldValue: field / weight;
        });
    }

    void map(
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            unsigned int processedChildSubtrees,
            std::ostream& fieldMap) const
    {
        std::vector<BoundingCube<N, real_type>> processedAreas;
        {
            BlockIndex childIndex;
            auto iChildSubtree = 0u;
            auto childSize = make_real(0.5)*subtreePos.size();
            do {
                if (processedChildSubtrees & (1 << iChildSubtree)) {
                    auto childMinPos = subtreePos.min() + ScalarOrMultiIndex<N,real_type>::fromMultiIndex(childIndex*childSize);
                    processedAreas.push_back(BoundingCube<N, real_type>(childMinPos, childSize));
                }
                ++iChildSubtree;
            }
            while (inc01MultiIndex(childIndex));
        }

        for (auto zone : m_meshProvider.zones(cache(), coordVars())) {
            switch (zone.elementType()) {
            case MeshElementType::Triangle:
                mapZone<MeshElementType::Triangle>(
                            weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas, fieldMap);
                break;
            case MeshElementType::Quad:
                mapZone<MeshElementType::Quad>(
                            weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas, fieldMap);
                break;
            case MeshElementType::Tetrahedron:
                mapZone<MeshElementType::Tetrahedron>(
                            weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas, fieldMap);
                break;
            case MeshElementType::Hexahedron:
                mapZone<MeshElementType::Hexahedron>(
                            weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas, fieldMap);
                break;
            }
        }
    }

private:
    const Metadata<N>& m_metadata;
    unsigned int m_fieldIndex;
    MeshProvider& m_meshProvider;

    mutable typename MeshProvider::Cache m_cache;
    typename MeshProvider::Cache cache() const
    {
        if (!m_cache)
            m_cache = m_meshProvider.makeCache();
        return m_cache;
    }

    mutable std::vector<unsigned int> m_coordVars;
    const std::vector<unsigned int>& coordVars() const
    {
        if (m_coordVars.empty()) {
            m_coordVars = m_meshProvider.coordinateVariables();
            if (m_coordVars.size() != N)
                throw std::runtime_error("Invalid dimension in the mesh file");
        }
        return m_coordVars;
    }
};

} // s3dmm
