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

namespace s3dmm {

namespace detail {

template<unsigned int N>
struct TreeBlockId
{
    unsigned int index;     // index in m_data.children
    unsigned int level;     // level, starting from 0 for the entire cube
    MultiIndex<N, unsigned int> location;    // index identifying block position within its level
    TreeBlockId() : index(0), level(0) {}
    TreeBlockId(unsigned int index, unsigned int level, const MultiIndex<N, unsigned int>& location) :
        index(index), level(level), location(location)
    {}
    bool operator<(const TreeBlockId<N>& that) const {
        return level == that.level? index < that.index: level < that.level;
    }
    bool operator==(const TreeBlockId<N>& that) const {
        return level == that.level && index == that.index;
    }
};

} // detail

} // s3dmm
