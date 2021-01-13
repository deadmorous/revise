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

#include <boost/range/algorithm/copy.hpp>

// #define S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_ALLOW_TIMERS

#ifdef S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_ALLOW_TIMERS

#define S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_GET_TIMERS(tp) \
    auto tp = timerPtrs(); \
    ScopedTimerUser timerUser(nullptr)

#define S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, timer) \
    timerUser.replace(tp.timer)

#else // S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_ALLOW_TIMERS

#define S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_GET_TIMERS(tp)
#define S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, timer)

#endif // S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_ALLOW_TIMERS

namespace s3dmm {

// Note: IField means "interpolated field",
// according to the algorithm used to compute field values at block tree nodes.
template<unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeIFieldFromFile : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    struct Timers
    {
        ScopedTimer initTimer;
        ScopedTimer elementNodesTimer;
        ScopedTimer elementBoxTimer;
        ScopedTimer elementBoxCheckTimer;
        ScopedTimer elementApproxInitTimer;
        ScopedTimer elementApproxRunTimer;
        ScopedTimer fieldTransformTimer;
        ScopedTimer otherOpTimer;
    };

    BlockTreeIFieldFromFile(
            const Metadata<N>& metadata,
            unsigned int fieldIndex,
            MeshProvider& meshProvider, const std::shared_ptr<Timers>& timers = std::shared_ptr<Timers>()
            ) :
        m_metadata(metadata),
        m_fieldIndex(fieldIndex),
        m_meshProvider(meshProvider),
        m_timers(timers)
    {
    }

private:

    struct TimerPtrs
    {
        ScopedTimer *initTimer = nullptr;
        ScopedTimer *elementNodesTimer = nullptr;
        ScopedTimer *elementBoxTimer = nullptr;
        ScopedTimer *elementBoxCheckTimer = nullptr;
        ScopedTimer *elementApproxInitTimer = nullptr;
        ScopedTimer *elementApproxRunTimer = nullptr;
        ScopedTimer *fieldTransformTimer = nullptr;
        ScopedTimer *otherOpTimer = nullptr;
    };

    TimerPtrs timerPtrs() const
    {
        auto tm = m_timers.get();
        if (tm)
            return {
                &tm->initTimer,
                &tm->elementNodesTimer,
                &tm->elementBoxTimer,
                &tm->elementBoxCheckTimer,
                &tm->elementApproxInitTimer,
                &tm->elementApproxRunTimer,
                &tm->fieldTransformTimer,
                &tm->otherOpTimer
            };
        else
            return TimerPtrs();
    }

    template<MeshElementType elementType>
    void processZone(
            real_type *fieldValues,
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            const typename MeshProvider::Zone& zone,
            const std::vector<BoundingCube<N, real_type>>& processedAreas) const
    {
        S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_GET_TIMERS(tp);
        S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, initTimer);

        auto maxLevel = subtreeRoot.level + subtreeNodes.maxDepth();
        auto& blockTree = m_metadata.blockTree();
        BoundingBox<N, real_type> subtreeBox;
        subtreeBox << subtreePos.min() << subtreePos.max();

        auto elements = zone.elements();

        auto gridSize = subtreePos.size() / (1 << subtreeNodes.maxDepth());
        const BlockIndex ione = BlockIndex::filled(1u);

        S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, otherOpTimer);
        for (auto it=elements.begin(); it!=elements.end(); ++it) {
            S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, elementNodesTimer);
            auto elementNodes = it.elementNodes();

            S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, elementBoxTimer);
            auto elementBox = elementBoundingBox<N>(elementNodes) & subtreeBox;

            S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, elementBoxCheckTimer);
            if (!elementBox.empty() &&
                    !std::any_of(processedAreas.begin(), processedAreas.end(), [&](const auto& bc) {
                        return bc.contains(elementBox);
                    }))
            {
                S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, otherOpTimer);
                auto minPos = ((elementBox.min()-subtreePos.min())/gridSize);
                auto maxPos = ((elementBox.max()-subtreePos.min())/gridSize);
                auto imin = ScalarOrMultiIndex<N, real_type>::toMultiIndex(minPos).template convertTo<unsigned int>();
                auto imax = ScalarOrMultiIndex<N, real_type>::toMultiIndex(maxPos).template convertTo<unsigned int>() + ione;

                S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, elementApproxInitTimer);
                auto elementApprox = makeElementApprox<N, elementType>(elementNodes);

                S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, otherOpTimer);
                subtreeNodes.walkIndexBoxNodes(imin, imax, [&](const NodeIndex& nodeIndex, std::size_t nodeNumber) {
                    if (weightValues[nodeNumber] > 0)
                        return;
                    auto nodePos = reinterpret_cast<const Vec<N,real_type>&>(subtreePos.min()) + nodeIndex*gridSize;
                    real_type fieldValue;

                    S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, elementApproxRunTimer);
                    if (elementApprox(&fieldValue, nodePos)) {
                        fieldValues[nodeNumber] += fieldValue;
                        ++weightValues[nodeNumber];
                    }
                });
            }
            S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, otherOpTimer);
        }
    }

public:
    void generate(
            real_type *fieldValues,
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            real_type noFieldValue,
            unsigned int processedChildSubtrees) const
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

        auto nodeCount = subtreeNodes.nodeCount();
        for (auto zone : m_meshProvider.zones(cache(), vars())) {
            switch (zone.elementType()) {
            case MeshElementType::Triangle:
                processZone<MeshElementType::Triangle>(
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas);
                break;
            case MeshElementType::Quad:
                processZone<MeshElementType::Quad>(
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas);
                break;
            case MeshElementType::Tetrahedron:
                processZone<MeshElementType::Tetrahedron>(
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas);
                break;
            case MeshElementType::Hexahedron:
                processZone<MeshElementType::Hexahedron>(
                            fieldValues, weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas);
                break;
            }
        }

        S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_GET_TIMERS(tp)
        S3DMM_BLOCK_TREE_I_FIELD_FROM_FILE_USE_TIMER(tp, fieldTransformTimer)
        const auto Tol = make_real(1e-5);
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
    std::shared_ptr<Timers> m_timers;

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
};

} // s3dmm
