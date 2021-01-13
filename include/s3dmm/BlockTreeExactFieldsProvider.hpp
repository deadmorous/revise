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
#include "foreach_byindex32.hpp"
#include "computeFieldRange.hpp"
#include "signed_shift.hpp"

#include <boost/range/algorithm/fill.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/algorithm/copy.hpp>

#include <experimental/filesystem>
#include <fstream>
#include <map>

namespace s3dmm {

template<unsigned int N>
class BlockTreeExactFieldsProvider : boost::noncopyable
{
public:
    enum ProgressStage { FieldMapStage, FieldGenerationStage };
    struct ProgressCallbackData
    {
        ProgressStage stage;
        unsigned int level;
        unsigned int metadataBlock;
        unsigned int metadataBlockCount;
    };
    using ProgressCallback = std::function<void(const ProgressCallbackData&)>;
    using BlockId = typename Metadata<N>::BlockId;
    using BlockIndex = typename Metadata<N>::BlockIndex;

    struct Timers
    {
        ScopedTimer fieldInitTimer;
        ScopedTimer blockTreeNodesTimer;
        ScopedTimer fieldReadTimer;
        ScopedTimer fieldPropagationTimer;
        ScopedTimer fieldGenerationTimer;
        ScopedTimer fieldTransformTimer;
        ScopedTimer fieldWriteTimer;
        ScopedTimer fieldMapGenerationTimer;
        ScopedTimer otherOpTimer;
    };

    template<class ExactFieldGenerator>
    BlockTreeExactFieldsProvider(
            const Metadata<N>& metadata,
            const std::string& fieldBaseName,
            const ExactFieldGenerator& fieldGenerator,
            const ProgressCallback& progressCallback = ProgressCallback(),
            const std::shared_ptr<Timers>& timers = std::shared_ptr<Timers>()
            ) :
        m_metadata(metadata),
        m_fieldBaseName(fieldBaseName),
        m_timers(timers)
    {
        using namespace std::experimental::filesystem;
        auto fieldNames = fieldGenerator.fieldNames();
        std::vector<unsigned int> missingFieldIndices;
        foreach_byindex32(ifield, fieldNames) {
            if (!exists(fieldFileName(fieldNames[ifield])))
                missingFieldIndices.push_back(ifield);
        }
        if (!missingFieldIndices.empty())
            generateFields(missingFieldIndices, fieldGenerator, progressCallback);
    }

    const Metadata<N>& metadata() const {
        return m_metadata;
    }

    static constexpr real_type noFieldValue() {
        return make_real(1e30);
    }

private:
    const Metadata<N>& m_metadata;
    std::string m_fieldBaseName;
    std::shared_ptr<Timers> m_timers;

    std::string fieldFileName(const std::string& fieldName) const {
        return m_fieldBaseName + ".s3dmm-field#" + fieldName;
    }

    template<class BT>
    std::vector<real_type> fieldValuesPriv(std::istream& is, const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        Vec2<real_type> fieldRange;
        std::vector<real_type> sparseField;
        fieldValuesPriv(is, fieldRange, sparseField, subtreeNodes);
        return sparseField;
    }

    template<class BT>
    void fieldValuesPriv(
            std::istream& is,
            Vec2<real_type>& fieldRange,
            std::vector<real_type>& sparseField,
            const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        sparseField.resize(subtreeNodes.nodeCount());
        auto sd = m_metadata.subtreeData(subtreeNodes.root());
        is.seekg(sd.subtreeValuesPos);
        is.read(reinterpret_cast<char*>(fieldRange.data()), 2*sizeof(real_type));
        is.read(reinterpret_cast<char*>(sparseField.data()), static_cast<std::streamsize>(sparseField.size()*sizeof(real_type)));
    }

    static void writeZeros(std::ostream& os, std::size_t byteCount)
    {
        const unsigned int BufSize = std::min(std::size_t(1024*1024), byteCount);
        std::vector<char> buf(BufSize, 0);
        auto remainingBytes = byteCount;
        while (remainingBytes != 0u) {
            auto bytesToWrite = std::min(buf.size(), remainingBytes);
            os.write(buf.data(), static_cast<std::streamoff>(bytesToWrite));
            if (os.fail())
                throw std::runtime_error("File write error");
            remainingBytes -= bytesToWrite;
        }
    }

    struct TimerPtrs
    {
        ScopedTimer *fieldInitTimer = nullptr;
        ScopedTimer *blockTreeNodesTimer = nullptr;
        ScopedTimer *fieldReadTimer = nullptr;
        ScopedTimer *fieldPropagationTimer = nullptr;
        ScopedTimer *fieldGenerationTimer = nullptr;
        ScopedTimer *fieldTransformTimer = nullptr;
        ScopedTimer *fieldWriteTimer = nullptr;
        ScopedTimer *fieldMapGenerationTimer = nullptr;
        ScopedTimer *otherOpTimer = nullptr;
    };

    TimerPtrs timerPtrs() const
    {
        auto tm = m_timers.get();
        if (tm)
            return {
                &tm->fieldInitTimer,
                &tm->blockTreeNodesTimer,
                &tm->fieldReadTimer,
                &tm->fieldPropagationTimer,
                &tm->fieldGenerationTimer,
                &tm->fieldTransformTimer,
                &tm->fieldWriteTimer,
                &tm->fieldMapGenerationTimer,
                &tm->otherOpTimer
            };
        else
            return TimerPtrs();
    }

    template<class ExactFieldGenerator>
    void generateFields(
            const std::vector<unsigned int>& fieldIndices,
            const ExactFieldGenerator& fieldGenerator,
            const ProgressCallback& progressCallback)
    {
        auto tp = timerPtrs();
        ScopedTimerUser timerUser(tp.fieldInitTimer);
        auto fieldCount = fieldIndices.size();
        std::vector<std::unique_ptr<std::fstream>> fieldStreams(fieldCount);
        auto fieldNames = fieldGenerator.fieldNames();
        for(auto ifield=0u; ifield<fieldCount; ++ifield) {
            auto ffn = fieldFileName(fieldNames[fieldIndices[ifield]]);
            auto& os = *(fieldStreams[ifield] = std::make_unique<std::fstream>(
                        ffn, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc));
            if (os.fail())
                throw std::runtime_error(std::string("Failed to open field file '") + ffn + " for writing");
        }
        auto fieldFileSize = m_metadata.fieldFileSize();
        for (auto& osp : fieldStreams)
            writeZeros(*osp, fieldFileSize);
        auto& blockTree = m_metadata.blockTree();
        std::vector<std::vector<real_type>> fieldValues(fieldCount);     // index=node number, value=field
        std::vector<std::vector<real_type>> weightValues(fieldCount);    // index=node number, value=weight
        std::vector<real_type> childFieldValues;
        auto levels = m_metadata.levels();
        auto levelCount = levels.size();
        auto metadataBlockCount = m_metadata.metadataBlockCount();
        auto imetadataBlock = 0u;

        timerUser.replace(tp.otherOpTimer);
        for (auto ilevel=levelCount-1; ilevel!=std::size_t(~0); --ilevel) {
            auto& level = levels[ilevel];
            for (auto metadataBlock : level) {
                if (progressCallback)
                    progressCallback({FieldGenerationStage, static_cast<unsigned int>(ilevel), imetadataBlock, metadataBlockCount});

                auto subtreeRoot = metadataBlock.subtreeRoot();

                timerUser.replace(tp.blockTreeNodesTimer);
                auto subtreeNodes = m_metadata.blockTreeNodes(subtreeRoot);
                auto subtreeDepth = subtreeNodes.maxDepth();
                auto nodeCount = subtreeNodes.nodeCount();

                timerUser.replace(tp.otherOpTimer);
                for(auto ifield=0u; ifield<fieldCount; ++ifield) {
                    auto& fv = fieldValues[ifield];
                    auto& fw = weightValues[ifield];
                    fv.resize(nodeCount);
                    fw.resize(nodeCount);
                    boost::range::fill(fv, make_real(0));
                    boost::range::fill(fw, make_real(0));
                }

                auto allSubtreesExist = false;
                if (ilevel+1 < levelCount) {
                    // Propagate field from higher level
                    BlockIndex childIndex;
                    auto subtreeCount = 0u;
                    auto iChildSubtree = 0u;
                    do {
                        auto childLocation = (subtreeRoot.location << 1) | childIndex;
                        auto maybeSubtreeData = m_metadata.maybeSubtreeData(ilevel+1, childLocation);
                        if (maybeSubtreeData.second) {
                            BlockId childSubtreeRoot(maybeSubtreeData.first.subtreeBlockTreePos, ilevel+1, childLocation);

                            timerUser.replace(tp.blockTreeNodesTimer);
                            auto childSubtreeNodes = m_metadata.blockTreeNodes(childSubtreeRoot);
                            auto childSubtreeDepth = childSubtreeNodes.maxDepth();
                            auto childNodeIndexOffset =
                                childIndex.template convertTo<BlockTreeNodeCoord>()
                                << (subtreeDepth-1);
                            int shift = childSubtreeDepth + 1 - subtreeDepth;
                            for(auto ifield=0u; ifield<fieldCount; ++ifield) {
                                timerUser.replace(tp.fieldReadTimer);
                                Vec2<real_type> childFieldRange;
                                fieldValuesPriv(*fieldStreams[ifield], childFieldRange, childFieldValues, childSubtreeNodes);
                                auto& fv = fieldValues[ifield];
                                auto& fw = weightValues[ifield];

                                timerUser.replace(tp.fieldPropagationTimer);
                                auto& childNodeData = childSubtreeNodes.data();
                                BOOST_ASSERT(childNodeData.n2i.size() == childFieldValues.size());
                                foreach_byindex32 (ichildNode, childFieldValues) {
                                    auto& childNodeIndex = childNodeData.n2i[ichildNode];
                                    if (shift < 1 || (childNodeIndex & 1).is_zero()) {
                                        auto nodeIndex = signed_shift_right(childNodeIndex, shift) + childNodeIndexOffset;
                                        auto inode = subtreeNodes.nodeNumber(nodeIndex);
                                        auto childValue = childFieldValues[ichildNode];
                                        if (childValue != noFieldValue()) {
                                            fv[inode] += childValue;
                                            ++fw[inode];
                                        }
                                    }
                                }
                            }

                            timerUser.replace(tp.otherOpTimer);
                            ++subtreeCount;
                        }
                        ++iChildSubtree;
                    }
                    while (inc01MultiIndex(childIndex));
                    allSubtreesExist = subtreeCount == (1 << N);
                }

                if (!allSubtreesExist) {
                    // Compute any missing field values
                    timerUser.replace(tp.fieldGenerationTimer);
                    auto subtreePos = blockTree.blockPos(subtreeRoot);
                    auto gridSize = subtreePos.size() / (1 << subtreeNodes.maxDepth());

                    std::vector<real_type> fieldBuf(fieldGenerator.fieldCount());
                    auto& subtreeNodeIndices = subtreeNodes.data().n2i;
                    for (auto inode=0u; inode<nodeCount; ++inode) {
                        auto nodeIndex = subtreeNodeIndices[inode];
                        if (weightValues[0][inode] <= 0) {
                            auto nodePos = reinterpret_cast<const Vec<N,real_type>&>(subtreePos.min()) + nodeIndex*gridSize;
                            fieldGenerator.fieldValues(fieldBuf.data(), nodePos);
                            for(auto ifield=0u; ifield<fieldCount; ++ifield) {
                                fieldValues[ifield][inode] += fieldBuf[fieldIndices[ifield]];
                                auto& w = weightValues[ifield][inode];
                                ++w;
                            }
                        }
                    }
                }

                // Normalize field values
                timerUser.replace(tp.fieldTransformTimer);
                for(auto ifield=0u; ifield<fieldCount; ++ifield) {
                    auto& fv = fieldValues[ifield];
                    auto& fw = weightValues[ifield];
                    std::transform(
                        fv.begin(), fv.end(), fw.begin(), fv.begin(),
                        [](real_type field, real_type weight)
                        {
                            return weight > 0? field / weight: noFieldValue();
                        });
                }

                // Write field values
                timerUser.replace(tp.fieldWriteTimer);
                for(auto ifield=0u; ifield<fieldCount; ++ifield) {
                    auto& os = *fieldStreams[ifield];
                    os.seekp(metadataBlock.subtreeData.subtreeValuesPos);
                    Vec2<real_type> fieldRange = computeFieldRange(fieldValues[ifield], noFieldValue());
                    os.write(reinterpret_cast<const char*>(fieldRange.data()), 2*sizeof(real_type));
                    os.write(reinterpret_cast<const char*>(fieldValues[ifield].data()), nodeCount*sizeof(real_type));
                }

                ++imetadataBlock;

                timerUser.replace(tp.otherOpTimer);
            }
        }
    }
};

} // s3dmm
