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

namespace s3dmm {

enum class BlockTreeFieldGenerationPolicy
{
    Separate,
    Propagate
};

template<unsigned int N>
class BlockTreeFieldProvider : boost::noncopyable
{
public:
    struct ProgressCallbackData
    {
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
        ScopedTimer otherOpTimer;
    };

    template<class FieldGenerator>
    BlockTreeFieldProvider(
            const Metadata<N>& metadata,
            const std::string& fieldFileName,
            const FieldGenerator& fieldGenerator,
            BlockTreeFieldGenerationPolicy generationPolicy,
            const ProgressCallback& progressCallback = ProgressCallback(),
            const std::shared_ptr<Timers>& timers = std::shared_ptr<Timers>()
            ) :
        m_metadata(metadata),
        m_fieldFileName(fieldFileName),
        m_timers(timers)
    {
        using namespace std::experimental::filesystem;
        if (!exists(m_fieldFileName)) {
            switch (generationPolicy) {
            case BlockTreeFieldGenerationPolicy::Separate:
                generateSeparatedField(fieldGenerator, progressCallback);
                break;
            case BlockTreeFieldGenerationPolicy::Propagate:
                generatePropagatedField(fieldGenerator, progressCallback);
                break;
            }
        }
        m_is.open(m_fieldFileName, std::ios::binary);
        if (m_is.fail())
            throw std::runtime_error("Failed to open field file for reading");
        m_is.exceptions(std::ifstream::failbit);
    }

    BlockTreeFieldProvider(
            const Metadata<N>& metadata,
            const std::string& fieldFileName) :
        m_metadata(metadata),
        m_fieldFileName(fieldFileName)
    {
        using namespace std::experimental::filesystem;
        if (!exists(m_fieldFileName))
            throw std::runtime_error(std::string("Field file '") + m_fieldFileName + "' does not exist");
        m_is.open(m_fieldFileName, std::ios::binary);
        if (m_is.fail())
            throw std::runtime_error("Failed to open field file for reading");
        m_is.exceptions(std::ifstream::failbit);
    }

    template<class BT>
    std::vector<real_type> fieldValues(const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        return fieldValuesPriv(m_is, subtreeNodes);
    }

    template<class BT>
    void fieldValues(Vec2<real_type>& fieldRange, std::vector<real_type>& sparseField, const BlockTreeNodes<N, BT>& subtreeNodes) {
        fieldValuesPriv(m_is, fieldRange, sparseField, subtreeNodes);
    }

    Vec2<real_type> fieldRange(const BlockId& root)
    {
        auto sd = m_metadata.subtreeData(root);
        m_is.seekg(sd.subtreeValuesPos);
        Vec2<real_type> result;
        m_is.read(reinterpret_cast<char*>(result.data()), 2*sizeof(real_type));
        return result;
    }

    const Metadata<N>& metadata() const {
        return m_metadata;
    }

    static constexpr real_type noFieldValue() {
        return make_real(1e30);
    }

private:
    const Metadata<N>& m_metadata;
    std::string m_fieldFileName;
    std::ifstream m_is;
    std::shared_ptr<Timers> m_timers;

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
                &tm->otherOpTimer
            };
        else
            return TimerPtrs();
    }

    template<class FieldGenerator>
    void generateSeparatedField(
            const FieldGenerator& fieldGenerator,
            const ProgressCallback& progressCallback)
    {
        auto tp = timerPtrs();
        ScopedTimerUser timerUser(tp.fieldInitTimer);
        std::ofstream os(m_fieldFileName, std::ios::binary);
        if (os.fail())
            throw std::runtime_error("Failed to open field file for writing");
        auto fieldFileSize = m_metadata.fieldFileSize();
        writeZeros(os, fieldFileSize);
        auto& blockTree = m_metadata.blockTree();
        std::vector<real_type> fieldValues;     // index=node number, value=field
        std::vector<real_type> weightValues;    // index=node number, value=weight
        auto metadataBlockCount = m_metadata.metadataBlockCount();
        auto ilevel = 0u;
        auto imetadataBlock = 0u;

        timerUser.replace(tp.otherOpTimer);
        for (auto level : m_metadata.levels()) {
            for (auto metadataBlock : level) {
                if (progressCallback)
                    progressCallback({ilevel, imetadataBlock, metadataBlockCount});

                auto subtreeRoot = metadataBlock.subtreeRoot();
                auto subtreePos = blockTree.blockPos(subtreeRoot);

                timerUser.replace(tp.blockTreeNodesTimer);
                auto subtreeNodes = m_metadata.blockTreeNodes(subtreeRoot);
                auto nodeCount = subtreeNodes.nodeCount();

                timerUser.replace(tp.otherOpTimer);
                fieldValues.resize(nodeCount);
                weightValues.resize(nodeCount);
                boost::range::fill(fieldValues, make_real(0));
                boost::range::fill(weightValues, make_real(0));

                timerUser.replace(tp.fieldGenerationTimer);
                fieldGenerator.generate(
                            fieldValues.data(), weightValues.data(),
                            subtreeRoot, subtreePos, subtreeNodes, noFieldValue(), 0);

                timerUser.replace(tp.fieldWriteTimer);
                os.seekp(metadataBlock.subtreeData.subtreeValuesPos);
                Vec2<real_type> fieldRange = computeFieldRange(fieldValues, noFieldValue());
                os.write(reinterpret_cast<const char*>(fieldRange.data()), 2*sizeof(real_type));
                os.write(reinterpret_cast<const char*>(fieldValues.data()), nodeCount*sizeof(real_type));

                timerUser.replace(tp.otherOpTimer);
                ++imetadataBlock;
            }
            ++ilevel;
        }
    }

    template<class FieldGenerator>
    void generatePropagatedField(
            const FieldGenerator& fieldGenerator,
            const ProgressCallback& progressCallback)
    {
        auto tp = timerPtrs();
        ScopedTimerUser timerUser(tp.fieldInitTimer);
        std::fstream os(m_fieldFileName, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (os.fail())
            throw std::runtime_error("Failed to open field file for writing");
        auto fieldFileSize = m_metadata.fieldFileSize();
        writeZeros(os, fieldFileSize);
        auto& blockTree = m_metadata.blockTree();
        std::vector<real_type> fieldValues;     // index=node number, value=field
        std::vector<real_type> weightValues;    // index=node number, value=weight
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
                    progressCallback({static_cast<unsigned int>(ilevel), imetadataBlock, metadataBlockCount});

                auto subtreeRoot = metadataBlock.subtreeRoot();
                auto subtreePos = blockTree.blockPos(subtreeRoot);

                timerUser.replace(tp.blockTreeNodesTimer);
                auto subtreeNodes = m_metadata.blockTreeNodes(subtreeRoot);
                auto subtreeDepth = subtreeNodes.maxDepth();
                auto nodeCount = subtreeNodes.nodeCount();

                timerUser.replace(tp.otherOpTimer);
                fieldValues.resize(nodeCount);
                weightValues.resize(nodeCount);
                boost::range::fill(fieldValues, make_real(0));
                boost::range::fill(weightValues, make_real(0));

                auto allSubtreesExist = false;
                auto processedChildSubtrees = 0u;
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

                            timerUser.replace(tp.fieldReadTimer);
                            Vec2<real_type> childFieldRange;
                            fieldValuesPriv(os, childFieldRange, childFieldValues, childSubtreeNodes);

                            timerUser.replace(tp.fieldPropagationTimer);
                            auto& childNodeData = childSubtreeNodes.data();
                            BOOST_ASSERT(childNodeData.n2i.size() == childFieldValues.size());
                            auto childNodeIndexOffset =
                                    childIndex.template convertTo<BlockTreeNodeCoord>()
                                    << (subtreeDepth-1);
                            int shift = childSubtreeDepth + 1 - subtreeDepth;
                            foreach_byindex32 (ichildNode, childFieldValues) {
                                auto& childNodeIndex = childNodeData.n2i[ichildNode];
                                if (shift < 1 || (childNodeIndex & 1).is_zero()) {
                                    auto nodeIndex = signed_shift_right(childNodeIndex, shift) + childNodeIndexOffset;
                                    auto inode = subtreeNodes.nodeNumber(nodeIndex);
                                    auto childValue = childFieldValues[ichildNode];
                                    if (childValue != noFieldValue()) {
                                        fieldValues[inode] += childValue;
                                        ++weightValues[inode];
                                    }
                                }
                            }

                            timerUser.replace(tp.otherOpTimer);
                            ++subtreeCount;
                            processedChildSubtrees |= 1 << iChildSubtree;
                        }
                        ++iChildSubtree;
                    }
                    while (inc01MultiIndex(childIndex));
                    allSubtreesExist = subtreeCount == (1 << N);
                }

                if (allSubtreesExist) {
                    timerUser.replace(tp.fieldTransformTimer);
                    std::transform(
                                fieldValues.begin(), fieldValues.end(), weightValues.begin(), fieldValues.begin(),
                                [](real_type field, real_type weight)
                    {
                        return weight > 0? field / weight: noFieldValue();
                    });
                }
                else {
                    timerUser.replace(tp.fieldGenerationTimer);
                    fieldGenerator.generate(
                                fieldValues.data(), weightValues.data(),
                                subtreeRoot, subtreePos, subtreeNodes, noFieldValue(), processedChildSubtrees);
                }

                timerUser.replace(tp.fieldWriteTimer);
                os.seekp(metadataBlock.subtreeData.subtreeValuesPos);
                Vec2<real_type> fieldRange = computeFieldRange(fieldValues, noFieldValue());
                os.write(reinterpret_cast<const char*>(fieldRange.data()), 2*sizeof(real_type));
                os.write(reinterpret_cast<const char*>(fieldValues.data()), nodeCount*sizeof(real_type));
                ++imetadataBlock;

                timerUser.replace(tp.otherOpTimer);
            }
        }
    }
};

} // s3dmm
