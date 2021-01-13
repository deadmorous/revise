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
class BlockTreeXWFieldFromImage : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeXWFieldFromImage(
            const Metadata<N>& metadata,
            const Func& imageFunc,
            const MultiIndex<N, unsigned int>& imageSize) :
        m_metadata(metadata),
        m_imageFunc(imageFunc),
        m_imageSize(imageSize)
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
        auto blockDepth = subtreeNodes.data().maxDepth;
        auto maxLevel = subtreeRoot.level + blockDepth;
        auto& blockTree = m_metadata.blockTree();
        using I2B = ImageToBlockIndexTransform<N>;
        auto irange = I2B::imageIndexRangeForBlock(m_imageSize, subtreeRoot);
        I2B i2b(m_imageSize, subtreeRoot, blockDepth, subtreePos);
        if (irange.first != irange.second) {
            auto iimage = irange.first;
            do {
                auto iblock = i2b(iimage);
                auto v = m_imageFunc(iimage);
                if (isnan(v))
                    continue;
                auto pos = i2b.pos(iimage);
                BOOST_ASSERT(subtreePos.contains(pos, tol));
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
            while(incMultiIndex(iimage, irange.first, irange.second));
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
    const Func& m_imageFunc;
    MultiIndex<N, unsigned int> m_imageSize;

    static bool isSameOrChildOf(const BlockId& maybeChild, const BlockId& maybeParent)
    {
        if (maybeChild.level < maybeParent.level)
            return false;
        auto loc = maybeChild.location >> (maybeChild.level - maybeParent.level);
        return loc == maybeParent.location;
    }
};

} // s3dmm
