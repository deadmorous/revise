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
#include <functional>

namespace std {

template<unsigned int N, class T>
struct hash<s3dmm::MultiIndex<N, T>>
{
    using argument_type = s3dmm::MultiIndex<N, T>;
    using result_type = size_t;
    result_type operator()(const argument_type& idx) const
    {
        hash<T> itemHash;
        const auto M = sizeof(size_t)*8/N;
        std::size_t result = 0u;
        for (auto d=0u; d<N; ++d)
            result = (result << M) ^ itemHash(idx[d]);
        return result;
    }
};

} // std
