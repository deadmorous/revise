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

template<unsigned int N>
inline std::size_t indexToOrdinal(const MultiIndex<N, unsigned int>& index, const MultiIndex<N, unsigned int>& size)
{
    std::size_t result = index[N-1];
    for (auto d=1; d<N; ++d)
        result = result*size[N-d] + index[N-d-1];
    return result;
}

template<>
inline std::size_t indexToOrdinal<1>(const MultiIndex<1, unsigned int>& index, const MultiIndex<1, unsigned int>&) {
    return static_cast<std::size_t>(index[0]);
}

template<>
inline std::size_t indexToOrdinal<2>(const MultiIndex<2, unsigned int>& index, const MultiIndex<2, unsigned int>& size) {
    return index[0] + static_cast<std::size_t>(size[0])*index[1];
}

template<>
inline std::size_t indexToOrdinal<3>(const MultiIndex<3, unsigned int>& index, const MultiIndex<3, unsigned int>& size) {
    return index[0] + static_cast<std::size_t>(size[0])*(index[1] + static_cast<std::size_t>(size[1])*index[2]);
}

} // s3dmm
