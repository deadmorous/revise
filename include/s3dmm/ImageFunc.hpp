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

#include "RectGridData.hpp"
#include "indexToOrdinal.hpp"

namespace s3dmm {

namespace detail {

template<class T>
struct Identity {
    T operator()(const T& x) const {
        return x;
    }
};

} // detail

template<unsigned int N, class Tout, class Tin = Tout, class Transform = detail::Identity<Tin>>
class ImageFunc
{
public:
    using Index = MultiIndex<N, unsigned int>;
    ImageFunc(
            const RectGridData<N> *grid,
            const std::string& arrayName,
            const Transform& transform = Transform()) :
        m_grid(grid),
        m_arrayName(arrayName),
        m_transform(transform)
    {}

    Tout operator()(const Index& pos) const
    {
        prepare();
        return m_transform(m_imageArr.at(indexToOrdinal(pos, m_imageSize)));
    }

private:
    const RectGridData<N> *m_grid;
    std::string m_arrayName;
    Transform m_transform;

    mutable Index m_imageSize;
    mutable std::vector<Tin> m_imageArr;
    void prepare() const
    {
        if (m_imageArr.empty()) {
            auto& imageData = m_grid->imageData();
            m_imageSize = imageData.imageSize();
            imageData.getArray(m_arrayName, m_imageArr);
        }
    }
};

} // s3dmm

