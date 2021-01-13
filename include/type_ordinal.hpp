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

#include <cstddef>

#include <boost/assert.hpp>

namespace s3dmm {

template<class T> struct type_ordinal;
template<class T> constexpr unsigned int type_ordinal_v = type_ordinal<T>::value;

template<> struct type_ordinal<bool> { static constexpr unsigned int value = 0; };
template<> struct type_ordinal<char> { static constexpr unsigned int value = 1; };
template<> struct type_ordinal<unsigned char> { static constexpr unsigned int value = 2; };
template<> struct type_ordinal<short> { static constexpr unsigned int value = 3; };
template<> struct type_ordinal<unsigned short> { static constexpr unsigned int value = 4; };
template<> struct type_ordinal<long> { static constexpr unsigned int value = 5; };
template<> struct type_ordinal<unsigned long> { static constexpr unsigned int value = 6; };
template<> struct type_ordinal<long long> { static constexpr unsigned int value = 7; };
template<> struct type_ordinal<unsigned long long> { static constexpr unsigned int value = 8; };
template<> struct type_ordinal<int> { static constexpr unsigned int value = 9; };
template<> struct type_ordinal<unsigned int> { static constexpr unsigned int value = 10; };
template<> struct type_ordinal<float> { static constexpr unsigned int value = 11; };
template<> struct type_ordinal<double> { static constexpr unsigned int value = 12; };

inline std::size_t type_size_by_ordinal(unsigned int ordinal) {
    switch (ordinal) {
    case type_ordinal_v<bool>:   return sizeof(bool);
    case type_ordinal_v<char>:   return sizeof(char);
    case type_ordinal_v<unsigned char>:   return sizeof(unsigned char);
    case type_ordinal_v<short>:   return sizeof(short);
    case type_ordinal_v<unsigned short>:   return sizeof(unsigned short);
    case type_ordinal_v<long>:   return sizeof(long);
    case type_ordinal_v<unsigned long>:   return sizeof(unsigned long);
    case type_ordinal_v<long long>:   return sizeof(long long);
    case type_ordinal_v<unsigned long long>:   return sizeof(unsigned long long);
    case type_ordinal_v<int>:   return sizeof(int);
    case type_ordinal_v<unsigned int>:   return sizeof(unsigned int);
    case type_ordinal_v<float>:   return sizeof(float);
    case type_ordinal_v<double>:   return sizeof(double);
    default:
        BOOST_ASSERT(false);
        return 0;
    }
}

} // s3dmm
