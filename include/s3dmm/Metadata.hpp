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
#include "BlockTreeNodes.hpp"
#include "BlockTree_io.hpp"
#include "BlockTreeNodes_io.hpp"
#include "PrefixWriter.hpp"

#include <boost/noncopyable.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/lexical_cast.hpp>

#include <ostream>

// #define S3DMM_STORE_BLOCKTREE_NODES
#define S3DMM_STORE_COMPRESSED_BLOCKTREE

#ifdef S3DMM_STORE_COMPRESSED_BLOCKTREE
#include "BlockTreeCompressor.hpp"
#define S3DMM_USE_COMPRESSED_BLOCKTREE
#endif // S3DMM_STORE_COMPRESSED_BLOCKTREE

/// \ingroup s3dmm
/// \brief The namespace for the s3dmm core library API
namespace s3dmm {

namespace detail {

struct Metadata_SubtreeData_v0
{
    unsigned int subtreeBlockTreePos = 0;
#ifdef S3DMM_STORE_BLOCKTREE_NODES
    std::size_t subtreeNodesPos = 0;
#endif // S3DMM_STORE_BLOCKTREE_NODES
    std::size_t subtreeValuesPos = 0;
};

struct Metadata_SubtreeData
{
    unsigned int subtreeBlockTreePos = 0;
#ifdef S3DMM_STORE_BLOCKTREE_NODES
    std::size_t subtreeNodesPos = 0;
#endif // S3DMM_STORE_BLOCKTREE_NODES
    std::size_t subtreeValuesPos = 0;
    std::size_t subtreeDepth = 0;

    static Metadata_SubtreeData from_v0(const Metadata_SubtreeData_v0& that)
    {
        Metadata_SubtreeData result;
        result.subtreeBlockTreePos = that.subtreeBlockTreePos;
#ifdef S3DMM_STORE_BLOCKTREE_NODES
        result.subtreeNodesPos = that.subtreeNodesPos;
#endif // S3DMM_STORE_BLOCKTREE_NODES
        result.subtreeValuesPos = that.subtreeValuesPos;
        return result;
    }
};

template <class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const Metadata_SubtreeData& sd)
{
    writer << sd.subtreeBlockTreePos
#ifdef S3DMM_STORE_BLOCKTREE_NODES
           << sd.subtreeNodesPos
#endif // S3DMM_STORE_BLOCKTREE_NODES
           << sd.subtreeValuesPos
           << sd.subtreeDepth;
    return writer;
}

template <class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, Metadata_SubtreeData_v0& sd)
{
    reader >> sd.subtreeBlockTreePos
#ifdef S3DMM_STORE_BLOCKTREE_NODES
           >> sd.subtreeNodesPos
#endif // S3DMM_STORE_BLOCKTREE_NODES
           >> sd.subtreeValuesPos;
    return reader;
}

template <class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
    BinaryReaderTemplate<S, VS, SS>& reader, Metadata_SubtreeData& sd)
{
    reader >> sd.subtreeBlockTreePos
#ifdef S3DMM_STORE_BLOCKTREE_NODES
        >> sd.subtreeNodesPos
#endif // S3DMM_STORE_BLOCKTREE_NODES
        >> sd.subtreeValuesPos
        >> sd.subtreeDepth;
    return reader;
}

} // detail

template<unsigned int N>
class Metadata : boost::noncopyable
{
public:
    using SubtreeData = detail::Metadata_SubtreeData;
#ifdef S3DMM_USE_COMPRESSED_BLOCKTREE
    using BT = CompressedBlockTree<N>;
#else // S3DMM_USE_COMPRESSED_BLOCKTREE
    using BT = BlockTree<N>;
#endif // S3DMM_USE_COMPRESSED_BLOCKTREE
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;

private:
    using LevelBlockMap = std::unordered_map<BlockIndex, SubtreeData>;

public:
    struct SubtreeNodesProgressCallbackData
    {
        unsigned int level;
        unsigned int subtree;
        unsigned int subtreeCount;
        const BlockTreeNodes<N, BT>& blockTreeNodes;
    };
    using SubtreeNodesProgressCallback = std::function<void(const SubtreeNodesProgressCallbackData&)>;

#ifdef S3DMM_USE_COMPRESSED_BLOCKTREE
    static BT convertBlockTree(const BlockTree<N>& bt) {
        return BlockTreeCompressor<N>::compressBlockTree(bt);
    }
#else // S3DMM_USE_COMPRESSED_BLOCKTREE
    static const BT& convertBlockTree(const BlockTree<N>& bt) { return bt; }
    static BT&& convertBlockTree(BlockTree<N>&& bt) { return std::move(bt); }
#endif // S3DMM_USE_COMPRESSED_BLOCKTREE

    Metadata() = default;

    Metadata(
            std::ostream& os,
            const BT& bt,
            unsigned int maxSubtreeDepth,
            unsigned int maxFullLevel,
            const SubtreeNodesProgressCallback& subtreeNodesProgressCallback = SubtreeNodesProgressCallback()) :
        m_blockTree(bt),
        m_maxSubtreeDepth(maxSubtreeDepth),
        m_maxFullLevel(maxFullLevel)
    {
        generate(os, subtreeNodesProgressCallback);
    }

    Metadata(
            std::ostream& os,
            BT&& bt,
            unsigned int maxSubtreeDepth,
            unsigned int maxFullLevel,
            const SubtreeNodesProgressCallback& subtreeNodesProgressCallback = SubtreeNodesProgressCallback()) :
        m_blockTree(std::move(bt)),
        m_maxSubtreeDepth(maxSubtreeDepth),
        m_maxFullLevel(maxFullLevel)
    {
        generate(os, subtreeNodesProgressCallback);
    }

    explicit Metadata(std::istream& is) : m_is(&is)
    {
        auto formatVersion = readHeader(is);
        BinaryReader reader(is);
        auto endPos = reader.read<std::size_t>();
        is.seekg(static_cast<std::streamoff>(endPos));
#if defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        CompressedBlockTree<N> cbt;
        reader >> cbt;
        BlockTreeCompressor<N>::decompressBlockTree(m_blockTree, cbt);
#else // defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        reader >> m_blockTree;
#endif // defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        if (formatVersion == 0) {
            reader >> m_maxSubtreeDepth;
            using LevelBlockMap_v0 = std::unordered_map<BlockIndex, detail::Metadata_SubtreeData_v0>;
            std::vector<LevelBlockMap_v0> subtreeData_v0;
            reader >> subtreeData_v0;
            for (auto& level_v0 : subtreeData_v0) {
                m_subtreeData.emplace_back();
                auto& level = m_subtreeData.back();
                for (auto& item_v0 : level_v0) {
                    auto& data = level[item_v0.first] = SubtreeData::from_v0(item_v0.second);
                    data.subtreeDepth = m_maxSubtreeDepth;
                }
            }
        }
        else {
            reader >> m_subtreeData;
        }
        reader >> m_fieldFileSize;
        initBlockTreeSearch();
    }

    BlockId closestSubtree(const BlockIndex& blockIndex, unsigned int level) const
    {
        BOOST_ASSERT(!m_subtreeData.empty());
        BlockIndex closestBlockIndex = blockIndex;
        auto maxLevel = static_cast<unsigned int>(m_subtreeData.size() - 1);
        if (level > maxLevel) {
            closestBlockIndex >>= level - maxLevel;
            level = maxLevel;
        }
        for (; level!=~0u; --level, closestBlockIndex>>=1) {
            auto& levelSubtreeData = m_subtreeData[level];
            auto it = levelSubtreeData.find(closestBlockIndex);
            if (it != levelSubtreeData.end())
                return { it->second.subtreeBlockTreePos, level, closestBlockIndex };
        }
        BOOST_ASSERT(false);
        return BlockId();
    }

    const BT& blockTree() const {
        return m_blockTree;
    }

    unsigned int maxSubtreeDepth() const {
        return m_maxSubtreeDepth;
    }

    unsigned int subtreeDepth(unsigned int level, const BlockIndex& blockIndex) const {
        return subtreeData(level, blockIndex).subtreeDepth;
    }

    unsigned int subtreeDepth(const BlockId& subtreeRoot) const {
        return subtreeData(subtreeRoot).subtreeDepth;
    }

    unsigned int maxFullLevel() const {
        return m_maxFullLevel;
    }

    BlockTreeNodes<N, BT> blockTreeNodes(const BlockId& subtreeRoot) const
    {
#ifdef S3DMM_STORE_BLOCKTREE_NODES
        BOOST_ASSERT(m_is);
        BinaryReader reader(*m_is);
        auto& subtreeData = m_subtreeData.at(subtreeRoot.level).at(subtreeRoot.location);
        m_is->seekg(subtreeData.subtreeNodesPos);
        BlockTreeNodes<N, BT> btn;
        reader >> btn;
        return std::move(btn);
#else // S3DMM_STORE_BLOCKTREE_NODES
        return BlockTreeNodes<N, BT>(m_blockTree, subtreeRoot, subtreeDepth(subtreeRoot));
#endif // S3DMM_STORE_BLOCKTREE_NODES
    }

    template<class It>
    using range = boost::iterator_range<It>;

    struct MetadataBlock
    {
        unsigned int level;
        BlockIndex blockIndex;
        SubtreeData subtreeData;
        BlockId subtreeRoot() const {
            return { subtreeData.subtreeBlockTreePos, level, blockIndex };
        }
    };

    class LevelIterator;

    class LevelBlocksIterator : public boost::iterator_facade<
            LevelBlocksIterator, const MetadataBlock, boost::forward_traversal_tag>
    {
    public:
        LevelBlocksIterator() = default;

        const MetadataBlock& dereference() const {
            return m_metadataBlock = { m_level, m_it->first, m_it->second };
        }

        void increment() {
            ++m_it;
        }

        bool equal(const LevelBlocksIterator& that) const {
            BOOST_ASSERT(m_level == that.m_level);
            return m_it == that.m_it;
        }

    private:
        unsigned int m_level;
        typename LevelBlockMap::const_iterator m_it;
        mutable MetadataBlock m_metadataBlock;
        LevelBlocksIterator(unsigned int level, const typename LevelBlockMap::const_iterator& it) :
            m_level(level), m_it(it)
        {}
        friend class LevelIterator;
    };
    using LevelBlocks = range<LevelBlocksIterator>;

    class LevelIterator : public boost::iterator_facade<
            LevelIterator, const LevelBlocks, boost::random_access_traversal_tag,
            const LevelBlocks>
    {
    public:
        const LevelBlocks dereference() const
        {
            auto& levelBlocks = m_subtreeData.at(m_level);
            return LevelBlocks(
                        LevelBlocksIterator(m_level, levelBlocks.begin()),
                        LevelBlocksIterator(m_level, levelBlocks.end()));
        }

        void increment() {
            ++m_level;
        }

        bool equal(const LevelIterator& that) const
        {
            BOOST_ASSERT(&m_subtreeData == &that.m_subtreeData);
            return  m_level == that.m_level;
        }

        void advance(std::ptrdiff_t distance) {
            m_level += distance;
        }

        std::ptrdiff_t distance_to(const LevelIterator& that) const
        {
            BOOST_ASSERT(&m_subtreeData == &that.m_subtreeData);
            return static_cast<std::ptrdiff_t>(that.m_level) - m_level;
        }

    private:
        const std::vector<LevelBlockMap>& m_subtreeData;
        unsigned int m_level;
        LevelIterator(const std::vector<LevelBlockMap>& subtreeData, unsigned int level) :
            m_subtreeData(subtreeData), m_level(level)
        {}
        friend class Metadata<N>;
    };
    using Levels = range<LevelIterator>;

    Levels levels() const {
        return Levels(LevelIterator(m_subtreeData, 0), LevelIterator(m_subtreeData, m_subtreeData.size()));
    }

    std::size_t fieldFileSize() const {
        return m_fieldFileSize;
    }

    unsigned int metadataBlockCount() const
    {
        auto result = 0u;
        for (auto& level : m_subtreeData)
            result += level.size();
        return result;
    }

    std::pair<SubtreeData, bool> maybeSubtreeData(unsigned int level, const BlockIndex& blockIndex) const
    {
        if (level < m_subtreeData.size()) {
            auto& levelBlocks = m_subtreeData[level];
            auto it = levelBlocks.find(blockIndex);
            if (it != levelBlocks.end())
                return { it->second, true };
        }
        return { SubtreeData(), false };
    }

    SubtreeData subtreeData(unsigned int level, const BlockIndex& blockIndex) const {
        return m_subtreeData.at(level).at(blockIndex);
    }

    SubtreeData subtreeData(const BlockId& blockId) const {
        return subtreeData(blockId.level, blockId.location);
    }

private:
    std::istream *m_is = nullptr;
    BT m_blockTree;
    unsigned int m_maxSubtreeDepth = 0;
    unsigned int m_maxFullLevel = 0;
    // index = level, value = (key = subtree root block index, value = subtree data)
    std::vector<LevelBlockMap> m_subtreeData;
    std::size_t m_fieldFileSize = 0;
    void generate(std::ostream& os, const SubtreeNodesProgressCallback& subtreeNodesProgressCallback)
    {
        initBlockTreeSearch();
        writeHeader(os);
        PrefixWriter prefixWriter(os);
        prefixWriter.allocPrefix(); // Later we will write the position of subtree data at this place
        BinaryWriter writer(os);
        std::size_t vpos = 0;
        auto isubtree = 0u;
        auto subtreeCount = 0u;
        m_blockTree.walkSubtrees(m_maxFullLevel, m_maxSubtreeDepth, [&](const BlockId&) {
            ++subtreeCount;
        });
        m_blockTree.walkSubtrees(m_maxFullLevel, m_maxSubtreeDepth, [&, this](const BlockId& subtreeRoot) {
            auto depth = m_blockTree.depthUpTo(subtreeRoot, m_maxSubtreeDepth);
            BlockTreeNodes<N, BT> btn(m_blockTree, subtreeRoot, depth);
#ifdef S3DMM_STORE_BLOCKTREE_NODES
            auto ndpos = static_cast<std::size_t>(os.tellp());
            writer << btn;
            SubtreeData sd = { subtreeRoot.index, ndpos, vpos, depth };
#else // S3DMM_STORE_BLOCKTREE_NODES
            SubtreeData sd = { subtreeRoot.index, vpos, depth };
#endif // S3DMM_STORE_BLOCKTREE_NODES
            if (m_subtreeData.size() < subtreeRoot.level+1)
                m_subtreeData.resize(subtreeRoot.level+1);
            m_subtreeData[subtreeRoot.level][subtreeRoot.location] = sd;
            // first 2 scalars are field min and max, then go field values
            vpos += (2 + btn.data().n2i.size()) * sizeof(real_type);
            if (subtreeNodesProgressCallback)
                subtreeNodesProgressCallback({
                    subtreeRoot.level, isubtree, subtreeCount, btn});
            ++isubtree;
        });
        m_fieldFileSize = vpos;
        prefixWriter.writePosPrefix();

#if defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        writer << BlockTreeCompressor<N>::compressBlockTree(m_blockTree);
#else // defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        writer << m_blockTree;
#endif // defined (S3DMM_STORE_COMPRESSED_BLOCKTREE) && !defined (S3DMM_USE_COMPRESSED_BLOCKTREE)
        writer << m_subtreeData << m_fieldFileSize;
    }

    void initBlockTreeSearch()
    {
#ifdef S3DMM_USE_COMPRESSED_BLOCKTREE
        m_blockTree.maybeInitSearch();
#endif // S3DMM_USE_COMPRESSED_BLOCKTREE
    }

    void writeHeader(std::ostream& os)
    {
        auto idString = fileTypeIdString();
        os.write(idString.data(), idString.size());
        BinaryWriter writer(os);
        writer << FileFormatVersion;
        PrefixWriter pw(os);
        pw.allocPrefix();
        writer << N << m_maxSubtreeDepth << m_maxFullLevel;
        pw.writeSizePrefix();
    }

    unsigned int readHeader(std::istream& is)
    {
        auto idString = fileTypeIdString();
        std::vector<char> id(idString.size()+1, 0);
        is.read(id.data(), idString.size());
        if (idString != id.data()) {
            is.seekg(0);
            m_maxFullLevel = 0;
            return 0;
        }
        else {
            BinaryReader reader(is);
            auto formatVersion = reader.read<unsigned int>();
            if (formatVersion > FileFormatVersion)
                throw std::runtime_error(
                    "Failed to read metadata: unsupported format version " +
                    boost::lexical_cast<std::string>(formatVersion));
            auto headerSize = reader.read<std::size_t>();
            auto pos = is.tellg();
            auto dim = reader.read<unsigned int>();
            if (dim != N)
                throw std::runtime_error(
                    "Failed to read metadata: wrong dimension " +
                    boost::lexical_cast<std::string>(dim));
            reader >> m_maxSubtreeDepth >> m_maxFullLevel;
            if (is.tellg() - pos != headerSize)
                throw std::runtime_error("Failed to read metadata: invalid header");
            return formatVersion;
        }
    }

    static std::string fileTypeIdString() {
        return "s3dmm-metadata";
    }

    static constexpr unsigned int FileFormatVersion = 1;
};

} // s3dmm
