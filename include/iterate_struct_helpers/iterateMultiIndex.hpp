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

#include "silver_bullets/iterate_struct/iterate_struct.hpp"

#include "./UpTo4DVectorFieldNames.hpp"

#include <boost/assert.hpp>

namespace silver_bullets {
namespace iterate_struct {

template <unsigned int N, class T>
class iterate_struct_helper<s3dmm::MultiIndex<N, T>>
{
public:
    using S = s3dmm::MultiIndex<N, T>;
    static auto asTuple(S& s) {
        return asTupleHelper(s, std::make_integer_sequence<unsigned int, N>());
    }
    static auto asTuple(const S& s) {
        return asTupleHelper(s, std::make_integer_sequence<unsigned int, N>());
    }

    BOOST_STATIC_ASSERT(N <= 4);
    static constexpr const char* const* fieldNames = UpTo4DVectorFieldNames::fieldNames;

private:
    template<class S_or_ConstS, unsigned int ... i>
    static auto asTupleHelper(S_or_ConstS& s, std::integer_sequence<unsigned int, i...>) {
        return std::forward_as_tuple(s[i] ...);
    }
};

} // namespace iterate_struct
} // namespace silver_bullets
