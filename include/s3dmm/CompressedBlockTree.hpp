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

#include "BoundingCube.hpp"
#include "TreeBlockId.hpp"
#include "BitPacker.hpp"
#include "BlockTree_defs.hpp"
#include <vector>

namespace s3dmm {

template<unsigned int N>
class CompressedBlockTree
{
public:
    static constexpr const unsigned int ChildCount = 1 << N;
    using CompressedChildren = BitPackContainer_t<ChildCount>;
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;
    using BlockIndex = MultiIndex<N, unsigned int>;
    using BlockId = detail::TreeBlockId<N>;

    MultiIndex<N, real_type> vertexPos(const BlockIndex& vertexIndex, unsigned int level) const
    {
        auto blockSize = m_data.bc.size() / (1<<level);
        auto blockRelOrigin = vertexIndex * blockSize;
        auto blockPos = m_data.bc.min();
        for (auto d=0u; d<N; ++d)
            ScalarOrMultiIndex<N, real_type>::element(blockPos, d) += blockRelOrigin[d];
        return blockPos;
    }

    BoundingCube<N, real_type> blockPos(const BlockId& blockId) const
    {
        auto blockSize = m_data.bc.size() / (1<<blockId.level);
        auto blockRelOrigin = blockId.location * blockSize;
        auto blockPos = m_data.bc.min();
        for (auto d=0u; d<N; ++d)
            ScalarOrMultiIndex<N, real_type>::element(blockPos, d) += blockRelOrigin[d];
        return BoundingCube<N, real_type>(blockPos, blockSize);
    }

    BoundingCube<N, real_type> rootPos() const {
        return m_data.bc;
    }

    BlockId blockAt(const BlockIndex& blockIndex, unsigned int level) const
    {
        auto index = 0u;
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        BOOST_ASSERT((blockIndex >> level) == BlockIndex());
        for (auto gen=0u; gen<level; ++gen) {
            auto childLocalIndex = 0u;
            for (auto d=N-1; d!=~0u; --d)
                childLocalIndex = (childLocalIndex << 1) + ((blockIndex[d] >> (level-1-gen)) & 1);
            auto children = packer.get(index);
            auto childIndex = getChildIndex(index, childLocalIndex);
            if (!childIndex)
                throw std::range_error("Block does not exist");
            index = childIndex;
        }
        return {index, level, blockIndex};
    }

    BlockId blockAt(const vector_type& pos) const
    {
        BOOST_ASSERT(m_data.bc.contains(pos, m_data.bc.size()*relTol()));
        BlockId result;
        auto bcPos = m_data.bc.min();
        auto bcHalfSize = make_real(0.5) * m_data.bc.size();
        while(true) {
            auto childLocalIndex = childLocalIndexByPoint(bcPos, bcHalfSize, pos);
            auto childIndex = getChildIndex(result.index, childLocalIndex);
            if (childIndex) {
                auto childLocalLocation = childLocalLocationByLocalIndex(childLocalIndex);
                bcPos += childLocalLocation*bcHalfSize;
                bcHalfSize *= make_real(0.5);
                result.index = childIndex;
                result.location = (result.location << 1) + childLocalLocation;
                ++result.level;
            }
            else
                break;
        }
        return result;
    }

    BlockId depthLimitedBlockAt(const vector_type& pos, unsigned int maxLevel) const
    {
        BOOST_ASSERT(m_data.bc.contains(pos, m_data.bc.size()*relTol()));
        BlockId result;
        auto bcPos = m_data.bc.min();
        auto bcHalfSize = make_real(0.5) * m_data.bc.size();
        while(result.level < maxLevel) {
            auto childLocalIndex = childLocalIndexByPoint(bcPos, bcHalfSize, pos);
            auto childIndex = getChildIndex(result.index, childLocalIndex);
            if (childIndex) {
                auto childLocalLocation = childLocalLocationByLocalIndex(childLocalIndex);
                ScalarOrMultiIndex<N, real_type>::each_indexed(bcPos, [&](real_type& bcPosElement, unsigned int i) {
                    bcPosElement += childLocalLocation[i]*bcHalfSize;
                });
                bcHalfSize *= make_real(0.5);
                result.index = childIndex;
                result.location = (result.location << 1) + childLocalLocation;
                ++result.level;
            }
            else
                break;
        }
        return result;
    }

    unsigned int levelForSize(real_type size) const
    {
        static real_type ilog2 = make_real(1)/make_real(log(2));
        BOOST_ASSERT(0 < size && size < m_data.bc.size());
        return static_cast<unsigned int>(log(m_data.bc.size() / size)*ilog2 + 0.5);
    }

    template<class Cb>
    void walk(const BlockId& root, const Cb& cb) const
    {
        if (!cb(root))
            return;
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        auto children = packer.get(root.index);
        if (!children)
            return;
        auto childIndex = static_cast<unsigned int>(getFirstChildIndex(root.index));    // TODO better
        auto childMask = 1 << (ChildCount-1);
        for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex, childMask>>=1) {
            if (children & childMask) {
                walk({childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)}, cb);
                ++childIndex;
            }
        }
    }

    template<class Cb>
    bool walk(const BlockId& root, unsigned int maxDepth, const Cb& cb) const
    {
        cb(root);
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        auto children = packer.get(root.index);
        auto result = false;
        if (!children)
            return result;
        auto childIndex = static_cast<unsigned int>(getFirstChildIndex(root.index));    // TODO better
        auto childMask = 1 << (ChildCount-1);
        for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex, childMask>>=1) {
            if (children & childMask) {
                if (maxDepth > 0)
                    result = walk(
                        {childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)},
                        maxDepth-1, cb) || result;
                else
                    result = true;
                ++childIndex;
            }
        }
        return result;
    }

    template<class Cb>
    void walk(const Cb& cb) const
    {
        walk(BlockId(), cb);
    }

    template<class Cb>
    void walkSubtrees(
            const BlockId& root,
            unsigned int minDepth,
            unsigned int maxDepth,
            const Cb& cb) const
    {
        cb(root);
        if (minDepth > 0 || walk(root, maxDepth, [](const BlockId&) {})) {
            // auto& children = m_data.children.at(root.index);
            auto packer = makeBitPacker<ChildCount>(m_data.data.data());
            auto childIndex = getFirstChildIndex(root.index);
            auto children = packer.get(root.index);
            BOOST_ASSERT(children);
            auto childMask = 1 << (ChildCount-1);
            for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex, childMask>>=1) {
                if (children & childMask) {
                    BlockId childBlockId = {childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)};
#ifdef S3DMM_BLOCKTREE_COUNT_EACH_SUBTREE_DEPTH
                    if (minDepth > 0 || isSubtreeDeeperThan(childBlockId, maxDepth-1))
#endif // S3DMM_BLOCKTREE_COUNT_EACH_SUBTREE_DEPTH
                        walkSubtrees(childBlockId, (minDepth>0? minDepth-1: 0), maxDepth, cb);
                    ++childIndex;
                }
            }
        }
    }

    template<class Cb>
    void walkSubtrees(
            const BlockId& root,
            unsigned int maxDepth,
            const Cb& cb) const
    {
        walkSubtrees(root, 0, maxDepth, cb);
    }

    template<class Cb>
    void walkSubtrees(
            unsigned int maxDepth,
            const Cb& cb) const
    {
        walkSubtrees(BlockId(), 0, maxDepth, cb);
    }

    template<class Cb>
    void walkSubtrees(
            unsigned int minDepth,
            unsigned int maxDepth,
            const Cb& cb) const
    {
        walkSubtrees(BlockId(), minDepth, maxDepth, cb);
    }

    unsigned int childCount(const BlockId& blockId) const
    {
        auto result = 0u;
        for (auto childIndex : m_data.children.at(blockId.index)) {
            if (childIndex != 0u)
                ++result;
        }
        return result;
    }

    bool hasPartOfAllChildren(const BlockId& blockId) const
    {
        auto n = childCount(blockId);
        return n > 0 && n < ChildCount;
    }

    bool hasAllChildren(const BlockId& blockId) const {
        return childCount(blockId) == ChildCount;
    }

    bool hasNoChildren(const BlockId& blockId) const {
        return childCount(blockId) == 0u;
    }

    bool isSubtreeDeeperThan(const BlockId& subtreeRoot, unsigned int level) const {
        return walk(subtreeRoot, level, [](const auto& /*blockId*/) {});
    }

    unsigned int depth(const BlockId& subtreeRoot) const
    {
        auto maxLevel = subtreeRoot.level;
        walk(subtreeRoot, [&maxLevel](const auto& blockId) {
            if (maxLevel < blockId.level)
                maxLevel = blockId.level;
            return true;
        });
        return maxLevel - subtreeRoot.level;
    }

    unsigned int depth() const {
        return depth(BlockId());
    }

    unsigned int depthUpTo(const BlockId& subtreeRoot, unsigned int maxDepth) const
    {
        auto maxLevel = subtreeRoot.level;
        auto maxLevelThreshold = subtreeRoot.level + maxDepth;
        walk(subtreeRoot, [&](const auto& blockId) {
            if (maxLevel < blockId.level)
                maxLevel = blockId.level;
            return maxLevel < maxLevelThreshold;
        });
        return maxLevel - subtreeRoot.level;
    }

    unsigned int depthUpTp(unsigned int maxDepth) const {
        return depthUpTp(BlockId(), maxDepth);
    }

    struct Data
    {
        BoundingCube<N, real_type> bc;
        std::size_t dataBitCount;
        std::vector<CompressedChildren> data;
    };

    const Data& data() const {
        return m_data;
    }

    Data& mutableData() {
        return m_data;
    }

    static CompressedBlockTree<N> fromData(const Data& data)
    {
        CompressedBlockTree<N> result;
        result.m_data = data;
        return result;
    }

    static CompressedBlockTree<N> fromData(Data&& data)
    {
        CompressedBlockTree<N> result;
        result.m_data = std::move(data);
        return result;
    }

    void maybeInitSearch()
    {
        if (m_milestones.empty() && !m_data.data.empty())
            generateMilestones();
    }

private:
    static constexpr real_type relTol() {
        return make_real(1e-4);
    }

    template<class T>
    static T clamp(const T& x, const T& min, const T& max) {
        return x < min? min: x > max? max: x;
    }

    static BlockIndex childBlockLocation(const BlockIndex& location, unsigned int childLocalIndex)
    {
        BlockIndex result = location << 1;
        for (auto d=0u; d<N; ++d, childLocalIndex>>=1)
            result[d] += childLocalIndex & 1;
        return result;
    }

    static unsigned int childLocalIndexByPoint(
            const vector_type& bcPos, real_type bcHalfSize, const vector_type& pos)
    {
        auto childLocalIndex = 0u;
        BlockIndex childLocation;
        for (auto d=N-1; d!=~0u; --d) {
            childLocalIndex <<= 1;
            auto& bcPosElement = ScalarOrMultiIndex<N, real_type>::element(bcPos, d);
            auto& posElement = ScalarOrMultiIndex<N, real_type>::element(pos, d);
            if (posElement > bcPosElement + bcHalfSize)
                ++childLocalIndex;
        }
        return childLocalIndex;
    }

    static BlockIndex childLocationByLocalIndex(const BlockIndex& parentLocation, unsigned int childLocalIndex)
    {
        auto result = parentLocation << 1;
        for (auto d=0u; d<N; ++d)
            result[d] += (childLocalIndex>>d) & 1;
        return result;
    }

    static BlockIndex childLocalLocationByLocalIndex(unsigned int childLocalIndex)
    {
        BlockIndex result;
        for (auto d=0u; d<N; ++d)
            result[d] += (childLocalIndex>>d) & 1;
        return result;
    }

    Data m_data;

    static constexpr const unsigned int L2NodesPerMilestone = 3;
    static constexpr const unsigned int NodesPerMilestone = 1 << L2NodesPerMilestone;
    static constexpr const std::size_t MilestoneMask = (1 << L2NodesPerMilestone) - 1;
    mutable std::vector<std::size_t> m_milestones;
    mutable std::size_t m_totalNodeCount = 0;
    void generateMilestones()
    {
        auto nodeCount = m_data.dataBitCount >> N;
        m_milestones.resize((nodeCount+NodesPerMilestone-1) >> L2NodesPerMilestone);
        std::size_t imilestone = 0;
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        unsigned int newIndex = 1;
        CachedBitCounter<ChildCount>::maybeInit();
        for (std::size_t inode=0; inode<nodeCount; ++inode) {
            if ((inode & MilestoneMask) == 0)
                m_milestones[imilestone++] = newIndex;
            newIndex += CachedBitCounter<ChildCount>::countOnes(packer.get(inode));
        }
        m_totalNodeCount = newIndex;
        BOOST_ASSERT(newIndex == nodeCount);
    }

    std::size_t getFirstChildIndex(std::size_t parentIndex) const
    {
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        auto imilestone = parentIndex >> L2NodesPerMilestone;
        BOOST_ASSERT(imilestone < m_milestones.size());
        auto result = m_milestones[imilestone];
        BOOST_ASSERT(result < m_totalNodeCount);
        for (auto index=imilestone<<L2NodesPerMilestone; index<parentIndex; ++index)
            result += CachedBitCounter<ChildCount>::countOnes(packer.get(index));
        return result;
    }

    std::size_t getChildIndex(std::size_t parentIndex, unsigned int childLocalIndex) const
    {
        auto packer = makeBitPacker<ChildCount>(m_data.data.data());
        auto children = packer.get(parentIndex);
        if (children & (1 << (ChildCount-childLocalIndex-1))) {
            auto result = getFirstChildIndex(parentIndex);
            result += CachedBitCounter<ChildCount>::countHighestOnes(children, childLocalIndex);
            return result;
        }
        else
            return 0;
    }
};

} // s3dmm
