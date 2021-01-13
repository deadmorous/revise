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

#include "BlockTreeFieldProvider.hpp"
#include "filename_util.hpp"
#include "DenseFieldInterpolator.hpp"
#include "SyncRecentItemCache.hpp"
#include "FieldTimestamps.hpp"

namespace s3dmm {

template<unsigned int N>
class BlockTreeFieldService
{
public:
    using BlockId = typename BlockTree<N>::BlockId;

    explicit BlockTreeFieldService(const std::string& baseName) :
        m_baseName(baseName),
        m_btnCache(m_cacheMutex)
    {
        readFieldInfoFile();
        openMetadataFile();
    }

    unsigned int fieldCount() const {
        return m_fieldInfo.size();
    }

    std::vector<std::string> fieldNames() const
    {
        std::vector<std::string> result(m_fieldInfo.size());
        std::transform(
                    m_fieldInfo.begin(), m_fieldInfo.end(), result.begin(),
                    [](const FieldInfo& fieldInfo)
        {
            return fieldInfo.fieldName;
        });
        return result;
    }

    unsigned int maybeFieldIndex(const std::string& fieldName) const
    {
        auto it = std::find_if(m_fieldInfo.begin(), m_fieldInfo.end(), [&fieldName](const FieldInfo& fieldInfo) {
            return fieldInfo.fieldName == fieldName;
        });
        if (it == m_fieldInfo.end())
            return ~0u;
        else
            return it - m_fieldInfo.begin();
    }

    unsigned int fieldIndex(const std::string& fieldName) const
    {
        auto result = maybeFieldIndex(fieldName);
        if (result == ~0u)
            throw std::invalid_argument(std::string("Field '") + fieldName + "' is not defined");
        else
            return result;
    }

    std::string fieldName(unsigned int fieldIndex) const {
        return m_fieldInfo.at(fieldIndex).fieldName;
    }

    Vec2<real_type> fieldRange(unsigned int fieldIndex) const {
        return m_fieldInfo.at(fieldIndex).fieldRange;
    }

    unsigned int timeFrameCount() const {
        return m_timeFrameCount;
    }

    void interpolate(
            Vec2<dfield_real>& fieldRange,
            std::vector<dfield_real>& denseField,
            unsigned int fieldIndex,
            unsigned int timeFrame,
            const BlockId& subtreeRoot) const
    {
        interpolateWith<DenseFieldInterpolator<N>>(fieldRange, denseField, fieldIndex, timeFrame, subtreeRoot);
    }

    BlockTreeNodeCoord maxDenseFieldVerticesPerEdge() const {
        return IndexTransform<N>::template verticesPerEdge<BlockTreeNodeCoord>(metadata().maxSubtreeDepth());
    }

    std::size_t maxDenseFieldSize() const {
        return static_cast<std::size_t>(IndexTransform<N>::vertexCount(metadata().maxSubtreeDepth()));
    }

    BlockTreeNodeCoord denseFieldVerticesPerEdge(const BlockId& subtreeRoot) const {
        return IndexTransform<N>::template verticesPerEdge<BlockTreeNodeCoord>(metadata().subtreeDepth(subtreeRoot));
    }

    std::size_t denseFieldSize(const BlockId& subtreeRoot) const {
        return static_cast<std::size_t>(IndexTransform<N>::vertexCount(metadata().subtreeDepth(subtreeRoot)));
    }

    template <class DenseInterpolator>
    void interpolateWith(
            Vec2<dfield_real>& fieldRange,
            typename DenseInterpolator::dense_field_container_ref denseField,
            unsigned int fieldIndex,
            unsigned int timeFrame,
            const BlockId& subtreeRoot,
            FieldTimestamps *timestamps = nullptr) const
    {
        auto& bt = m_metadata->blockTree();
        const auto& fieldName = m_fieldInfo.at(fieldIndex).fieldName;
        auto fieldFileName = frameOutputFileName(m_baseName, timeFrame, m_hasTimeSteps) + ".s3dmm-field#" + fieldName;
        BlockTreeFieldProvider<N> btf(*m_metadata, fieldFileName);
        DenseInterpolator dfi(btf);
        dfi.setTimers(m_interpolatorTimers);
        auto& subtreeNodes = m_btnCache.get(subtreeRoot, [&]() {
            ScopedTimerUser blockTreeNodesTimerUser(m_interpolatorTimers? &m_interpolatorTimers.get()->blockTreeNodesTimer: nullptr);
            auto subtreeNodes = m_metadata->blockTreeNodes(subtreeRoot);
            return subtreeNodes;
        });
        Vec2<real_type> fieldRangeR;
        dfi.interpolate(fieldRangeR, denseField, subtreeNodes);
#ifdef S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        if (timestamps)
            *timestamps = dfi.timestamps();
#endif // S3DMM_ENABLE_WORKER_TIME_ESTIMATION
        fieldRange = fieldRangeR.convertTo<dfield_real>();
    }

    std::shared_ptr<DenseFieldInterpolatorTimers> interpolatorTimers() const {
        return m_interpolatorTimers;
    }
    void setInterpolatorTimers(const std::shared_ptr<DenseFieldInterpolatorTimers>& interpolatorTimers) {
        m_interpolatorTimers = interpolatorTimers;
    }

    const Metadata<N>& metadata() const {
        return *m_metadata;
    }

private:
    std::string m_baseName;
    std::unique_ptr<Metadata<N>> m_metadata;
    struct FieldInfo
    {
        std::string fieldName;
        Vec2<real_type> fieldRange;
        bool animated;
    };
    std::vector<FieldInfo> m_fieldInfo;
    unsigned int m_timeFrameCount = 0;
    bool m_hasTimeSteps = false;
    std::shared_ptr<DenseFieldInterpolatorTimers> m_interpolatorTimers;

    std::mutex m_cacheMutex;
    mutable SyncRecentItemCache<
        BlockId,
        BlockTreeNodes<N, typename Metadata<N>::BT>,
        std::mutex> m_btnCache;

    void readFieldInfoFile()
    {
        using namespace std;
        using namespace experimental::filesystem;
        string mainFileName;
        tie(mainFileName, m_hasTimeSteps) = firstOutputFrameFileName(m_baseName);
        string infoFileName;
        if (m_hasTimeSteps) {
            using namespace experimental::filesystem;
            auto s = splitFileName(m_baseName);
            infoFileName = path(get<0>(s)).append(get<1>(s)).append(get<1>(s) + get<2>(s) + ".s3dmm-fields");
        }
        else
            infoFileName = m_baseName + ".s3dmm-fields";
        ifstream is(infoFileName);
        if (is.fail())
            throw runtime_error(string("Failed to open the s3dmm info file '") + infoFileName + "'");
        string line;

        auto readPredefinedLine = [&](unsigned int lineNumber, const char *content)
        {
            getline(is, line);
            if (line != content) {
                ostringstream oss;
                oss << "Unexpected content of line "
                    << lineNumber
                    << " of file '"
                    << infoFileName + "':\nexpected '"
                    << content << "'\nfound '"
                    << line << "'";
                throw runtime_error(oss.str());
            }
        };

        auto readLine = [&](unsigned int lineNumber, bool required, auto f)
        {
            if (is.eof()) {
                if (required) {
                    ostringstream oss;
                    oss << "Unexpected end of file in line "
                        << lineNumber
                        << " of file '"
                        << infoFileName + "'";
                    throw runtime_error(oss.str());
                }
                else
                    return false;
            }
            getline(is, line);
            if (line.empty() && is.eof() && !required)
                return false;
            istringstream iss(line);
            iss.exceptions(ios::failbit);
            try {
                f(iss);
                return true;
            }
            catch(const exception& e) {
                ostringstream oss;
                oss << "In line "
                    << lineNumber
                    << " of file '"
                    << infoFileName + "': "
                    << e.what();
                throw runtime_error(oss.str());
            }
        };

        readPredefinedLine(1, "time_steps");
        readLine(2, true, [this](istream& s) {
            s >> m_timeFrameCount;
        });
        readPredefinedLine(3, "");
        readPredefinedLine(4, "field\tmin\tmax\tanimated");

        auto readFieldInfoLine = [this](istream& s) {
            string fieldName;
            real_type fmin;
            real_type fmax;
            int animated;
            s >> fieldName >> fmin >> fmax >> animated;
            m_fieldInfo.push_back({fieldName, {fmin, fmax}, animated != 0});
        };

        for (auto iline=5u; readLine(iline, false, readFieldInfoLine); ++iline) {}
    }

    void openMetadataFile()
    {
        using namespace std;
        string mainFileName = frameOutputFileName(m_baseName, 0, m_hasTimeSteps);
        auto metadataFileName = mainFileName + ".s3dmm-meta";
        ifstream is (metadataFileName, ios::binary);
        if (is.fail())
            throw runtime_error(string("Failed to open metadata file '") + metadataFileName + "'");
        m_metadata = make_unique<Metadata<N>>(is);
    }
};

} // s3dmm
