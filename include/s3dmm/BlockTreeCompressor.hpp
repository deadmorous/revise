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
#include "CompressedBlockTree.hpp"
#include <boost/static_assert.hpp>

namespace s3dmm {

template <unsigned int N>
class BlockTreeCompressor
{
private:
    static constexpr const unsigned int BitsPerNode = 1 << N;
    BOOST_STATIC_ASSERT(BitsPerNode >= 1 && BitsPerNode <= 8);
    static unsigned char packBlockTreeNodeChildren(const MultiIndex<BitsPerNode, unsigned int>& children)
    {
        unsigned char result = 0;
        for (auto child : children)
            result = (result << 1) | (child? 1: 0);
        return result;
    }

    static MultiIndex<BitsPerNode, unsigned int> unpackBlockTreeNodeChildren(unsigned char children, unsigned int& newIndex)
    {
        MultiIndex<BitsPerNode, unsigned int> result;
        unsigned char mask = 1 << (BitsPerNode-1);
        for (unsigned int i=0; i<BitsPerNode; ++i, mask >>= 1) {
            if (children & mask)
                result[i] = newIndex++;
        }
        return result;
    }

public:
    static CompressedBlockTree<N> compressBlockTree(const BlockTree<N>& bt)
    {

        auto& btData = bt.data();
        typename CompressedBlockTree<N>::Data cbtData;
        cbtData.bc = btData.bc;
        constexpr unsigned int BitsPerNode = 1 << N;
        auto nodeCount = btData.children.size();
        cbtData.dataBitCount = nodeCount << N;
        cbtData.data.resize((cbtData.dataBitCount + 7) >> 3);
        auto packer = makeBitPacker<BitsPerNode>(cbtData.data.data());
        for (std::size_t inode=0; inode<nodeCount; ++inode)
            packer.set(inode, packBlockTreeNodeChildren(btData.children[inode]));
        return CompressedBlockTree<N>::fromData(std::move(cbtData));
    }

    static void decompressBlockTree(BlockTree<N>& result, const CompressedBlockTree<N>& cbt)
    {
        auto& btData = result.mutableData();
        auto& cbtData = cbt.data();
        btData.bc = cbtData.bc;
        constexpr unsigned int BitsPerNode = 1 << N;
        auto nodeCount = cbtData.dataBitCount >> N;
        btData.children.resize(nodeCount);
        auto packer = makeBitPacker<BitsPerNode>(cbtData.data.data());
        unsigned int newIndex = 1;
        for (std::size_t inode=0; inode<nodeCount; ++inode)
            btData.children[inode] = unpackBlockTreeNodeChildren(packer.get(inode), newIndex);
        BOOST_ASSERT(newIndex == btData.children.size());
    }

    static BlockTree<N> decompressBlockTree(const CompressedBlockTree<N>& cbt)
    {
        BlockTree<N> result;
        decompressBlockTree(result, cbt);
        return result;
    }
};

} // s3dmm
