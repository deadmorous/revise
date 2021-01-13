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
#include "ImageFunc.hpp"
#include "ImageToBlockIndexTransform.hpp"
#include "IncMultiIndex.hpp"

#include <boost/range/algorithm/copy.hpp>

namespace s3dmm {

// Note: XWField means "extrapolated weighed field",
// according to the algorithm used to compute field values at block tree nodes.
// Note: The Func template parameter may be ImageFunc
template<unsigned int N, class Func>
class BlockTreeFieldFromFunc : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeFieldFromFunc(
            const Metadata<N>& metadata,
            const Func& func) :
        m_metadata(metadata),
        m_func(func)
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
        auto blockDepth = subtreeNodes.data().maxDepth;
        auto& blockTree = m_metadata.blockTree();
        blockTree.walk(subtreeRoot, blockDepth, [&, this](const BlockId& blockId) {
            auto blockPos = blockTree.blockPos(blockId);
            subtreeNodes.walkBlockNodes(blockId, [&, this](
                                                     unsigned int localNodeNumber,
                                                     const NodeIndex& /*nodeIndex*/,
                                                     std::size_t nodeNumber) {
                auto pos = blockPos.min();
                for (auto d=0u; d<N; ++d)
                    if ((localNodeNumber>>d) & 1)
                        ScalarOrMultiIndex<N, real_type>::element(pos, d) += blockPos.size();
                auto v = m_func(pos);
                if (!isnan(v)) {
                    fieldValues[nodeNumber] += v;
                    ++weightValues[nodeNumber];
                }
            });
        });
        std::transform(
                    fieldValues, fieldValues+subtreeNodes.nodeCount(),
                    weightValues,
                    fieldValues,
                    [noFieldValue](real_type field, real_type weight)
        {
            return weight <= 0? noFieldValue: field / weight;
        });
    }

private:
    const Metadata<N>& m_metadata;
    const Func& m_func;

    static bool isSameOrChildOf(const BlockId& maybeChild, const BlockId& maybeParent)
    {
        if (maybeChild.level < maybeParent.level)
            return false;
        auto loc = maybeChild.location >> (maybeChild.level - maybeParent.level);
        return loc == maybeParent.location;
    }
};

} // s3dmm
