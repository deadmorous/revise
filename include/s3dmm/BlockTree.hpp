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

template <unsigned int N> struct CompressedBlockTree;

template<unsigned int N>
class BlockTree
{
public:
    static constexpr const unsigned int ChildCount = 1<<N;
    using Children = MultiIndex<ChildCount, unsigned int>;
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;
    using BlockIndex = MultiIndex<N, unsigned int>;
    using BlockId = detail::TreeBlockId<N>;

    BlockTree() :
        m_data{ BoundingCube<N, real_type>(), {Children()} }
    {}

    explicit BlockTree(const BoundingCube<N, real_type>& bc) :
        m_data{ bc, {Children()} }
    {}

    explicit BlockTree(const BoundingBox<N, real_type>& bb) :
      m_data{ BoundingCube<N,real_type>(bb), {Children()} }
    {}

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

    BlockId ensureChildAt(const BlockId& blockId, const vector_type& pos)
    {
        auto& c = m_data.children.at(blockId.index);
        auto bc = blockPos(blockId);
        auto bcHalfSize = make_real(0.5) * bc.size();
        auto& bcPos = bc.min();
        auto childLocalIndex = childLocalIndexByPoint(bcPos, bcHalfSize, pos);
        // Note: avoid the seduction to assign m_data.children[blockId.index] to a reference-typed local variable
        if (!m_data.children[blockId.index][childLocalIndex]) {
            m_data.children[blockId.index][childLocalIndex] = m_data.children.size();
            m_data.children.push_back(Children());
        }
        return {
            m_data.children[blockId.index][childLocalIndex],
            blockId.level + 1,
            childLocationByLocalIndex(blockId.location, childLocalIndex)
        };
    }

    BlockId ensureChildAt(const BlockId& blockId, const BlockIndex& childLocalLocation)
    {
        auto& c = m_data.children.at(blockId.index);
        auto childLocalIndex = 0u;
        for (auto d=N-1; d!=~0u; --d) {
            BOOST_ASSERT(childLocalLocation[d] <= 1);
            childLocalIndex = (childLocalIndex << 1) + childLocalLocation[d];
        }
        auto& blockChildren = m_data.children[blockId.index];
        auto childBlockIndex = blockChildren[childLocalIndex];
        if (!childBlockIndex) {
            blockChildren[childLocalIndex] = childBlockIndex = m_data.children.size();
            m_data.children.push_back(Children());
        }
        return {
            childBlockIndex,
            blockId.level + 1,
            blockId.location + childLocalLocation
        };
    }

    BlockId ensureBlockAt(const vector_type& pos, unsigned int level)
    {
        BlockId blockId;
        for (auto gen=0u; gen<level; ++gen)
            blockId = ensureChildAt(blockId, pos);
        return blockId;
    }

    BlockId ensureBlockAt(const BlockIndex& blockIndex, unsigned int level)
    {
        BOOST_ASSERT((blockIndex >> level) == BlockIndex());
        BlockId blockId;
        for (auto gen=0u; gen<level; ++gen)
            blockId = ensureChildAt(blockId, (blockIndex >> (level-1-gen)) & 1);
        return blockId;
    }

    BlockId ensureBlockAt(const BoundingCube<N, real_type>& bc)
    {
        BOOST_ASSERT(m_data.bc.contains(bc, bc.size()*relTol()));
        return ensureBlockAt(bc.center(), levelForSize(m_data.bc.size()));
    }

    BlockId blockAt(const BlockIndex& blockIndex, unsigned int level) const
    {
        auto index = 0u;
        BOOST_ASSERT((blockIndex >> level) == BlockIndex());
        for (auto gen=0u; gen<level; ++gen) {
            auto childLocalIndex = 0u;
            for (auto d=N-1; d!=~0u; --d)
                childLocalIndex = (childLocalIndex << 1) + ((blockIndex[d] >> (level-1-gen)) & 1);
            auto childIndex = m_data.children[index][childLocalIndex];
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
            auto childIndex = m_data.children[result.index][childLocalIndex];
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
            auto childIndex = m_data.children[result.index][childLocalIndex];
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
        auto& children = m_data.children.at(root.index);
        for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex) {
            auto childIndex = children[childLocalIndex];
            if (childIndex)
                walk({childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)}, cb);
        }
    }

    template<class Cb>
    bool walk(const BlockId& root, unsigned int maxDepth, const Cb& cb) const
    {
        cb(root);
        auto& children = m_data.children.at(root.index);
        auto result = false;
        for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex) {
            auto childIndex = children[childLocalIndex];
            if (childIndex) {
                if (maxDepth > 0)
                    result = walk(
                        {childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)},
                        maxDepth-1, cb) || result;
                else
                    result = true;
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
            unsigned int maxDepth,
            const Cb& cb) const
    {
        cb(root);
        auto hasMoreChildren = walk(root, maxDepth, [](const BlockId&) {});
        if (hasMoreChildren) {
            auto& children = m_data.children.at(root.index);
            for (auto childLocalIndex=0u; childLocalIndex<ChildCount; ++childLocalIndex) {
                auto childIndex = children[childLocalIndex];
                if (childIndex) {
                    BlockId childBlockId = {childIndex, root.level+1, childBlockLocation(root.location, childLocalIndex)};
#ifdef S3DMM_BLOCKTREE_COUNT_EACH_SUBTREE_DEPTH
                    if (isSubtreeDeeperThan(childBlockId, maxDepth-1))
#endif // S3DMM_BLOCKTREE_COUNT_EACH_SUBTREE_DEPTH
                        walkSubtrees(childBlockId, maxDepth, cb);
                }
            }
        }
    }

    template<class Cb>
    void walkSubtrees(
            unsigned int maxDepth,
            const Cb& cb) const
    {
        walkSubtrees(BlockId(), maxDepth, cb);
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

    void makeCompressible()
    {
        if (!m_data.children.empty()) {
            std::vector<Children> children(m_data.children.size());
            auto count = static_cast<unsigned int>(m_data.children.size());
            children[0] = m_data.children[0];
            auto inew = 1u;
            for (auto i=0u; i<count; ++i) {
                auto& c = children[i];
                for (auto& ichild : c) {
                    if (ichild) {
                        children[inew] = m_data.children[ichild];
                        ichild = inew;
                        ++inew;
                    }
                }
            }
            BOOST_ASSERT(inew == count);
            m_data.children.swap(children);
        }
    }

    struct Data
    {
        BoundingCube<N, real_type> bc;
        std::vector<Children> children;
    };

    const Data& data() const {
        return m_data;
    }

    Data& mutableData() {
        return m_data;
    }

    static BlockTree<N> fromData(const Data& data)
    {
        BlockTree<N> result;
        result.m_data = data;
        return result;
    }

    static BlockTree<N> fromData(Data&& data)
    {
        BlockTree<N> result;
        result.m_data = std::move(data);
        return result;
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
};

} // s3dmm
