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
#include "ProgressReport.hpp"

#include "silver_bullets/fs_ns_workaround.hpp"

#include <fstream>

namespace s3dmm {

template <unsigned int N, class BlockTreeGenerator>
class MetadataProvider : boost::noncopyable
{
public:
    struct BlockTreeProgressCallbackData
    {
        unsigned int zone;
        unsigned int zoneCount;
        unsigned int zoneElement;
        unsigned int zoneElementCount;
        unsigned int blockTreeSize;
    };
    using SubtreeNodesProgressCallbackData = typename Metadata<N>::SubtreeNodesProgressCallbackData;
    MetadataProvider(
            const std::string& metadataFileName,
            unsigned int maxSubtreeDepth,
            unsigned int maxLevel,
            unsigned int maxFullLevel,
            const BlockTreeGenerator& blockTreeGenerator)
        :
        m_metadataFileName(metadataFileName),
        m_maxSubtreeDepth(maxSubtreeDepth),
        m_maxLevel(maxLevel),
        m_maxFullLevel(maxFullLevel),
        m_blockTreeGenerator(blockTreeGenerator)
    {
        BOOST_ASSERT(m_maxLevel + m_maxSubtreeDepth >= m_blockTreeGenerator.maxTreeDepth());
    }

    void setSubtreeNodesProgressCallback(
        const std::function<void(const SubtreeNodesProgressCallbackData&)>& subtreeNodesProgressCallback)
    {
        m_subtreeNodesProgressCallback = subtreeNodesProgressCallback;
    }

    Metadata<N>& metadata() const
    {
        using namespace std::experimental::filesystem;
        if (!m_metadata) {
            if (!exists(m_metadataFileName)) {
                std::ofstream os(m_metadataFileName, std::ios::binary);
                if (os.fail())
                    throw std::runtime_error("Failed to open output metadata file");
                REPORT_PROGRESS_STAGES();
                REPORT_PROGRESS_STAGE("Generate global block tree");
                auto bt = m_blockTreeGenerator.makeBlockTree();
#ifdef S3DMM_STORE_COMPRESSED_BLOCKTREE
                REPORT_PROGRESS_STAGE("Make global block tree compressible");
                bt.makeCompressible();
#endif // S3DMM_STORE_COMPRESSED_BLOCKTREE
                REPORT_PROGRESS_STAGE("Generate subtree nodes");
                Metadata<N> metadataGenerator(
                            os,
                            Metadata<N>::convertBlockTree(std::move(bt)),
                            m_maxSubtreeDepth,
                            m_maxFullLevel,
                            m_subtreeNodesProgressCallback);
            }
            m_is.open(m_metadataFileName, std::ios::binary);
            if (m_is.fail())
                throw std::runtime_error("Failed to open metadata file");
            m_metadata = std::make_unique<Metadata<N>>(m_is);
        }
        return *m_metadata.get();
    }

    const BlockTreeGenerator& blockTreeGenerator() const {
        return m_blockTreeGenerator;
    }

private:
    using vector_type = typename BoundingBox<N, real_type>::vector_type;

    std::string m_metadataFileName;
    unsigned int m_maxSubtreeDepth;
    unsigned int m_maxLevel;
    unsigned int m_maxFullLevel;
    BlockTreeGenerator m_blockTreeGenerator;

    std::function<void(const SubtreeNodesProgressCallbackData&)> m_subtreeNodesProgressCallback;
    mutable std::unique_ptr<Metadata<N>> m_metadata;
    mutable std::ifstream m_is;
};

} // s3dmm
