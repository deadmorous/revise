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
#include "foreach_byindex32.hpp"
#include "MultiIndex_hash.hpp"
#include "ProgressReport.hpp"
#include "IncMultiIndex.hpp"
#include "BlockTreeNodeCoord.hpp"

#include <unordered_map>

#define S3DMM_BLOCK_TREE_NODES_TIMING

#ifdef S3DMM_BLOCK_TREE_NODES_TIMING
#include <boost/timer/timer.hpp>

inline s3dmm::ProgressReport::times timesToSeconds(const boost::timer::cpu_times& t) {
    auto s = [](boost::timer::nanosecond_type ns) {
        return static_cast<s3dmm::ProgressReport::time_type>(ns) / 1e9;
    };
    return {s(t.wall), s(t.user), s(t.system)};
}

#define S3DMM_BLOCK_TREE_NODES_DECL_TIMER(timerName) boost::timer::cpu_timer timerName
#define S3DMM_BLOCK_TREE_NODES_COUNT_TIME(timerName, resultName) resultName = timesToSeconds(timerName.elapsed())
#else // S3DMM_BLOCK_TREE_NODES_TIMING
#define S3DMM_BLOCK_TREE_NODES_DECL_TIMER(timerName)
#define S3DMM_BLOCK_TREE_NODES_COUNT_TIME(timerName, resultName)
#endif // S3DMM_BLOCK_TREE_NODES_TIMING

namespace s3dmm {

template<unsigned int N, class BlockTree>
class BlockTreeNodes
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;
    using BT = BlockTree;
    using BlockId = typename BT::BlockId;
    using BlockIndex = MultiIndex<N, unsigned int>;
    using NodeCoord = BlockTreeNodeCoord;
    static constexpr const NodeCoord VertexCount = 1<<N;
    using NodeIndex = MultiIndex<N, NodeCoord>;

    BlockTreeNodes() = default;

    BlockTreeNodes(const BT& bt, const BlockId& root, unsigned int maxDepth) :
        m_data{root, maxDepth}
    {
        S3DMM_BLOCK_TREE_NODES_DECL_TIMER(timer);
        bt.walk(root, maxDepth, [this](const BlockId& blockId) {
            walkBlockNodesPriv(blockId, [this](unsigned int /*localNodeNumber*/, const NodeIndex& nodeIndex) {
                if (m_i2n.find(nodeIndex) == m_i2n.end())
                {
                    m_i2n[nodeIndex] = m_data.n2i.size();
                    m_data.n2i.push_back(nodeIndex);
                }
            });
        });
        S3DMM_BLOCK_TREE_NODES_COUNT_TIME(timer, m_generationTime);
    }

    template<class Cb>
    void walkBlockNodes(const BlockId& blockId, const Cb& cb) const
    {
        walkBlockNodesPriv(blockId, [&cb, this](unsigned int localNodeNumber, const NodeIndex& nodeIndex) {
            cb(localNodeNumber, nodeIndex, m_i2n.at(nodeIndex));
        });
    }

    // Note: imax is not inclusive!
    template<class Cb>
    void walkIndexBoxNodes(const NodeIndex& imin, const NodeIndex& imax, const Cb& cb) const
    {
        NodeIndex nodeIndex = imin;
        do {
            auto it = m_i2n.find(nodeIndex);
            if (it != m_i2n.end())
                cb(nodeIndex, it->second);
        }
        while (incMultiIndex(nodeIndex, imin, imax));
    }

    // Note: imax is not inclusive!
    template<class Cb>
    void walkIndexBoxNodesCancellable(const NodeIndex& imin, const NodeIndex& imax, const Cb& cb) const
    {
        NodeIndex nodeIndex = imin;
        do {
            auto it = m_i2n.find(nodeIndex);
            if (it != m_i2n.end()) {
                if (!cb(nodeIndex, it->second))
                    break;
            }
        }
        while (incMultiIndex(nodeIndex, imin, imax));
    }

    std::size_t nodeNumber(const NodeIndex& nodeIndex) const {
        return m_i2n.at(nodeIndex);
    }

    std::size_t nodeCount() const {
        return m_data.n2i.size();
    }

    unsigned int maxDepth() const {
        return m_data.maxDepth;
    }

    BlockId root() const {
        return m_data.root;
    }

    struct Data
    {
        BlockId root;
        unsigned int maxDepth;
        std::vector<NodeIndex> n2i;
    };

    const Data& data() const {
        return m_data;
    }

    static BlockTreeNodes<N, BT> fromData(const Data& data)
    {
        BlockTreeNodes<N, BT> result;
        result.m_data = data;
        result.generateI2n();
        return result;
    }

    static BlockTreeNodes<N, BT> fromData(Data&& data)
    {
        BlockTreeNodes<N, BT> result;
        result.m_data = std::move(data);
        result.generateI2n();
        return result;
    }

    const ProgressReport::times& generationTime() const {
        return m_generationTime;
    }

private:
    template<class Cb>
    void walkBlockNodesPriv(const BlockId& blockId, const Cb& f) const
    {
        BOOST_ASSERT(blockId.level >= m_data.root.level);
        auto blockRelLevel = blockId.level - m_data.root.level;
        BOOST_ASSERT(blockRelLevel <= m_data.maxDepth);
        auto levelShift = m_data.maxDepth - blockRelLevel;
        auto nodeIndexMask = (1u << m_data.maxDepth) - 1;
        auto nodeIndexBase = ((blockId.location << levelShift) & nodeIndexMask).template convertTo<NodeCoord>();
        for (NodeCoord localNodeNumber=0; localNodeNumber<VertexCount; ++localNodeNumber) {
            auto nodeIndex = nodeIndexBase;
            for (auto d=0u; d<N; ++d)
                nodeIndex[d] += ((localNodeNumber >> d) & 1) << levelShift;
            f(localNodeNumber, nodeIndex);
        }
    }

    Data m_data;
    std::unordered_map<NodeIndex, std::size_t> m_i2n;
    ProgressReport::times m_generationTime;

    void generateI2n()
    {
        m_i2n.reserve(m_data.n2i.size());
        foreach_byindex32 (nodeNumber, m_data.n2i)
            m_i2n[m_data.n2i[nodeNumber]] = nodeNumber;
    }
};

} // s3dmm
