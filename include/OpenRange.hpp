/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021-2024 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

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
#include "real_type.hpp"

#include <iostream>

namespace s3dmm
{

template <typename T>
struct OpenRange : MultiIndex<2, T>
{
    using Base = MultiIndex<2, T>;
    using Base::Base;

    bool empty() const noexcept
    { return (*this)[0] == (*this)[1]; }

    bool is_normalized() const noexcept
    { return (*this)[0] < (*this)[1]; }

    OpenRange normalized() const noexcept
    {
        if (is_normalized())
            return *this;
        else
            return { (*this)[1], (*this)[0] };
    }

    void normalize() noexcept
    { *this = normalized(); }

    T& origin() noexcept
    { return (*this)[0]; }

    const T& origin() const noexcept
    { return (*this)[0]; }

    T length() const noexcept
    { return (*this)[1] - (*this)[0]; }

    bool operator==(const OpenRange& that) const noexcept
    {
        return empty()
                   ? that.empty()
                   : static_cast<const Base&>(*this) == that;
    }

    static OpenRange make_empty() noexcept
    { return {0, 0}; }

    static OpenRange from_vec(const Base& vec) noexcept
    { return {vec[0], vec[1]}; }
};

template <typename T>
std::ostream& operator<<(std::ostream& s, const OpenRange<T>& r)
{
    if (r.empty())
        s << "empty";
    else
        s << '[' << r[0] << ',' << r[1] << ')';
    return s;
}

using IndexRange = OpenRange<unsigned int>;
using CoordRange = OpenRange<real_type>;

} // namespace s3dmm
