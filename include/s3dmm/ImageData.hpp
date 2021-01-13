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

#include "MultiIndex.hpp"
#include "type_ordinal.hpp"

#include <vector>
#include <map>
#include <functional>

#include <boost/any.hpp>
#include <boost/range/algorithm/find_if.hpp>
#include <boost/range/algorithm/transform.hpp>

namespace s3dmm {

template<unsigned int N>
class ImageData
{
public:
    using Index = MultiIndex<N, unsigned int>;

    struct ArrayInfo
    {
        unsigned int type;  // See s3dmm::type_ordinal
        std::size_t size;
    };

    Index gridSize() const {
        return m_gridSize;
    }

    Index imageSize() const {
        return m_gridSize - Index::filled(1);
    }

    unsigned int arrayCount() const {
        return static_cast<unsigned int>(m_arrayInfo.size());
    }

    std::vector<std::string> arrayNames() const
    {
        std::vector<std::string> result(m_arrayInfo.size());
        boost::range::transform(m_arrayInfo, result.begin(), [](auto& aa) { return aa.first; });
        return result;
    }

    unsigned int arrayId(const std::string& arrayName) const
    {
        auto it = boost::range::find_if(m_arrayInfo, [&](const auto& aa) {
            return aa.first == arrayName;
        });
        if (it == m_arrayInfo.end())
            throw std::range_error("Failed to find array '" + arrayName + "'");
        return static_cast<unsigned int>(it - m_arrayInfo.begin());
    }

    std::string arrayName(unsigned int arrayId) const {
        return m_arrayInfo.at(arrayId).first;
    }

    ArrayInfo arrayInfo(unsigned int arrayId) const {
        return m_arrayInfo.at(arrayId).second;
    }

    ArrayInfo arrayInfo(const std::string& arrayName) const {
        return arrayInfo(arrayId(arrayName));
    }

    std::vector<std::pair<std::string, ArrayInfo>> arrayInfo() const {
        return m_arrayInfo;
    }

    template<class T>
    void getArray(unsigned int arrayId, std::vector<T>& dst) const
    {
        using G = std::function<void(unsigned int, std::vector<T>&)>;
        auto& g = boost::any_cast<const G&>(m_arrayGetters.at(type_ordinal_v<T>));
        g(arrayId, dst);
    }

    template<class T>
    void getArray(const std::string& arrayName, std::vector<T>& dst) const {
        getArray<T>(arrayId(arrayName), dst);
    }

    void setGridSize(const Index& gridSize) {
        m_gridSize = gridSize;
    }

    void setArrayInfo(const std::vector<std::pair<std::string, ArrayInfo>>& arrayInfo) {
        m_arrayInfo = arrayInfo;
    }

    template<class T>
    void setArrayGetter(const std::function<void(unsigned int, std::vector<T>&)>& arrayGetter) {
        m_arrayGetters[type_ordinal_v<T>] = arrayGetter;
    }

private:
    Index m_gridSize;
    std::vector<std::pair<std::string, ArrayInfo>> m_arrayInfo;
    std::map<unsigned int, boost::any> m_arrayGetters;
};

} // s3dmm
