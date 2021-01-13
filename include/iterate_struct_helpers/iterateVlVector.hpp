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

#include "vlCore/Vector2.hpp"
#include "vlCore/Vector3.hpp"
#include "vlCore/Vector4.hpp"

#include "silver_bullets/iterate_struct/iterate_struct.hpp"

#include "./UpTo4DVectorFieldNames.hpp"

namespace silver_bullets {
namespace iterate_struct {

template <class T>
class iterate_struct_helper<vl::Vector2<T>>
{
public:
    using S = vl::Vector2<T>;
    static auto asTuple(S& s)
    {
        return std::forward_as_tuple(s[0], s[1]);
    }
    static auto asTuple(const S& s)
    {
        return std::forward_as_tuple(s[0], s[1]);
    }
    static constexpr const char* const* fieldNames = UpTo4DVectorFieldNames::fieldNames;
};

template <class T>
class iterate_struct_helper<vl::Vector3<T>>
{
public:
    using S = vl::Vector3<T>;
    static auto asTuple(S& s)
    {
        return std::forward_as_tuple(s[0], s[1], s[2]);
    }
    static auto asTuple(const S& s)
    {
        return std::forward_as_tuple(s[0], s[1], s[2]);
    }
    static constexpr const char* const* fieldNames = UpTo4DVectorFieldNames::fieldNames;
};

template <class T>
class iterate_struct_helper<vl::Vector4<T>>
{
public:
    using S = vl::Vector4<T>;
    static auto asTuple(S& s)
    {
        return std::forward_as_tuple(s[0], s[1], s[2], s[3]);
    }
    static auto asTuple(const S& s)
    {
        return std::forward_as_tuple(s[0], s[1], s[2], s[3]);
    }
    static constexpr const char* const* fieldNames = UpTo4DVectorFieldNames::fieldNames;
};

} // namespace iterate_struct
} // namespace silver_bullets

