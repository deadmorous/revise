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

#include "ImageData.hpp"
#include "real_type.hpp"

#include "binary_io.hpp"
#include "MultiIndex_binary_io.hpp"

#include <fstream>

namespace s3dmm {

namespace detail
{

template<unsigned int N>
struct RectGridDataArrayInfo : ImageData<N>::ArrayInfo
{
    std::string name;
    void read(BinaryReader& reader) {
        reader >> name >> this->type >> this->size;
    }

    void write(BinaryWriter& writer) const {
        writer << name << this->type << this->size;
    }
};

} // detail

template<unsigned int N>
class RectGridData
{
public:
    using Index = MultiIndex<N, unsigned int>;

    RectGridData() {
        m_is.exceptions(std::ios::failbit);
    }

    explicit RectGridData(const std::string& fileName) :
        m_is(fileName, std::ios::binary)
    {
        if (!m_is.is_open())
            throw std::runtime_error("Failed to open input file '" + fileName + "'");
        BinaryReader reader(m_is);
        auto fileSignature = reader.read<std::string>();
        if (fileSignature != m_fileSignature)
            throw std::runtime_error("Input file '" + fileName + "' is invalid or corrupt");
        auto version = reader.read<unsigned int>();
        if (version > m_version)
            throw std::runtime_error("Input file '" + fileName + "' has version newer than current");
        auto gridDimension = reader.read<unsigned int>();
        if (gridDimension != N)
            throw std::runtime_error(
                "Input file '" + fileName + "' contains a grid of dimension " +
                std::to_string(gridDimension) + ", while dimension " +
                std::to_string(N) + " was expected");
        auto gridSize = reader.read<Index>();
        reader >> m_gridCoordinates >> m_arrayInfo >> m_filePos;

        std::vector<std::pair<std::string, typename ImageData<N>::ArrayInfo>> imageArrayInfo(m_arrayInfo.size());
        boost::range::transform(m_arrayInfo, imageArrayInfo.begin(), [](auto& info) {
            return std::pair<std::string, typename ImageData<N>::ArrayInfo>{
                info.name, { info.type, info.size }
            };
        });

        m_imageData.setGridSize(gridSize);
        m_imageData.setArrayInfo(imageArrayInfo);

        m_imageData.template setArrayGetter<int>([this](unsigned int arrayId, std::vector<int>& dst) {
            getArray<int>(arrayId, dst);
        });
        m_imageData.template setArrayGetter<float>([this](unsigned int arrayId, std::vector<float>& dst) {
            getArray<float>(arrayId, dst);
        });
        m_imageData.template setArrayGetter<double>([this](unsigned int arrayId, std::vector<double>& dst) {
            getArray<double>(arrayId, dst);
        });
    }

    const ImageData<N>& imageData() const {
        return m_imageData;
    }

    const MultiIndex<N, std::vector<real_type>>& gridCoordinates() const {
        return m_gridCoordinates;
    }

    const std::vector<real_type>& gridCoordinates(unsigned int d) const {
        return m_gridCoordinates.at(d);
    }

    void setGridSize(const Index& gridSize) {
        m_imageData.setGridSize(gridSize);
    }

    void setGridCoordinates(const MultiIndex<N, std::vector<real_type>>& gridCoordinates) {
        m_gridCoordinates = gridCoordinates;
    }

    void setGridCoordinates(unsigned int d, const std::vector<real_type>& gridCoordinates) {
        m_gridCoordinates.at(d) = gridCoordinates;
    }

    template<class T>
    void addArrayInfo(const std::string& name, std::size_t size)
    {
        if (m_is.is_open() || m_os.is_open())
            throw std::runtime_error("addArrayInfo() failed: both input and output streams must be closed");
        m_arrayInfo.push_back({{ type_ordinal_v<T>, size }, name });
    }

    void openForWrite(const std::string& fileName)
    {
        if (m_is.is_open() || m_os.is_open())
            throw std::runtime_error("addArrayInfo() failed: both input and output streams must be closed");
        m_os.open(fileName, std::ios::binary);
        if (m_os.fail())
            throw std::runtime_error("Failed to open output file '" + fileName + "'");
        BinaryWriter writer(m_os);
        writer << std::string(m_fileSignature) << m_version
               << N << m_imageData.gridSize() << m_gridCoordinates << m_arrayInfo;
        m_filePos.resize(m_arrayInfo.size());
        if (!m_filePos.empty())
            m_filePos[0] = static_cast<int64_t>(m_os.tellp()) + sizeof(std::size_t) + m_filePos.size()*sizeof(std::int64_t);
        for (std::size_t i=1, n=m_arrayInfo.size(); i<n; ++i) {
            auto& info = m_arrayInfo[i-1];
            m_filePos[i] = m_filePos[i-1] + info.size * type_size_by_ordinal(info.type);
        }
        writer << m_filePos;
        BOOST_ASSERT(m_filePos.empty() || m_filePos[0] == m_os.tellp());
        m_arraysWritten = 0;
    }

    template<class T>
    void writeArray(const std::vector<T>& src)
    {
        if (!m_os.is_open())
            throw std::runtime_error("addArrayInfo() failed: output stream must be open");
        auto& info = m_arrayInfo.at(m_arraysWritten);
        if (info.type != type_ordinal_v<T>)
            throw std::runtime_error("addArrayInfo() failed: array has an invalid type");
        if (info.size != src.size())
            throw std::runtime_error("addArrayInfo() failed: array has an invalid size");
        m_os.write(reinterpret_cast<const char*>(src.data()), src.size()*sizeof(T));
        ++m_arraysWritten;
        if (m_arraysWritten == m_arrayInfo.size())
            m_os.close();
    }

    template<class T, class Range>
    void writeArray(const Range& src)
    {
        if (!m_os.is_open())
            throw std::runtime_error("addArrayInfo() failed: output stream must be open");
        auto& info = m_arrayInfo.at(m_arraysWritten);
        if (info.type != type_ordinal_v<T>)
            throw std::runtime_error("addArrayInfo() failed: array has an invalid type");
        if (info.size != src.size())
            throw std::runtime_error("addArrayInfo() failed: array has an invalid size");
        for (auto& v : src) {
            T vt = v;
            m_os.write(reinterpret_cast<const char*>(&vt), sizeof(T));
        }
        ++m_arraysWritten;
        if (m_arraysWritten == m_arrayInfo.size())
            m_os.close();
    }

private:
    ImageData<N> m_imageData;
    MultiIndex<N, std::vector<real_type>> m_gridCoordinates;
    std::ifstream m_is;
    std::ofstream m_os;
    unsigned int m_arraysWritten = 0;
    std::vector<detail::RectGridDataArrayInfo<N>> m_arrayInfo;
    std::vector<std::int64_t> m_filePos;
    static constexpr const char *m_fileSignature = "RectGridData_6810e83e-3d8c-4ee4-974e-3def6bfa6497";
    static constexpr unsigned int m_version = 1;

    template<class T>
    void getArray(unsigned int arrayId, std::vector<T>& dst)
    {
        auto& info = m_arrayInfo.at(arrayId);
        dst.resize(info.size);
        m_is.seekg(static_cast<std::ios::pos_type>(m_filePos.at(arrayId)));
        m_is.read(reinterpret_cast<char*>(dst.data()), info.size*sizeof(T));
    }
};

} // s3dmm
