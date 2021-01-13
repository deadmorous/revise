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
#include "real_type.hpp"

#include <type_traits>
#include <functional>
#include <cmath>
#include <numeric>

namespace s3dmm {

template<unsigned int N, class T = real_type> using Vec = MultiIndex<N, T>;

template< class T = real_type> using Vec2 = MultiIndex<2, T>;
template< class T = real_type> using Vec3 = MultiIndex<3, T>;

using Vec2d = MultiIndex<2, real_type>;
using Vec3d = MultiIndex<3, real_type>;
using Vec2f = MultiIndex<2, float>;
using Vec3f = MultiIndex<3, float>;
using Vec2i = MultiIndex<2, int>;
using Vec3i = MultiIndex<3, int>;
using Vec2u = MultiIndex<2, unsigned int>;
using Vec3u = MultiIndex<3, unsigned int>;

namespace VecPrivate {

template<class T1, class T2>
using common = typename std::common_type<T1, T2>::type;

template<unsigned int N, class T1, class T2>
using VecCommon = Vec<N, typename std::common_type<T1, T2>::type>;

} // VecPrivate


template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> operator+(const Vec<N, T1>& a, const Vec<N, T2>& b)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> operator-(const Vec<N, T1>& a, const Vec<N, T2>& b)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());
    return result;
}

template<unsigned int N, class T>
inline Vec<N, T> operator-(const Vec<N, T>& a)
{
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [](auto x) { return -x; });
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::common<T1, T2> operator*(const Vec<N, T1>& a, const Vec<N, T2>& b)
{
    auto result = VecPrivate::common<T1, T2>();
    for (unsigned int i=0; i<N; ++i)
        result += a[i]*b[i];
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> operator*(const Vec<N, T1>& a, const T2& b)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x * b; });
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> operator*(const T1& a, const Vec<N, T2>& b)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(b.begin(), b.end(), result.begin(), [&](auto x) { return a * x; });
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> operator/(const Vec<N, T1>& a, const T2& b)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x / b; });
    return result;
}

template<unsigned int N, class T1, class T2, class P>
inline VecPrivate::VecCommon<N, T1, T2> elementwise(const Vec<N, T1>& a, const Vec<N, T2>& b, P predicate)
{
    using T = VecPrivate::common<T1, T2>;
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), [&](auto ai, auto bi) { return predicate(ai, bi); });
    return result;
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> elementwiseMultiply(const Vec<N, T1>& a, const Vec<N, T2>& b)
{
    using T = VecPrivate::common<T1, T2>;
    return elementwise(a, b, std::multiplies<T>());
}

template<unsigned int N, class T1, class T2>
inline VecPrivate::VecCommon<N, T1, T2> elementwiseDivide(const Vec<N, T1>& a, const Vec<N, T2>& b)
{
    using T = VecPrivate::common<T1, T2>;
    return elementwise(a, b, std::divides<T>());
}

template<class T1, class T2>
inline Vec<3, VecPrivate::common<T1, T2>> operator%(const Vec<3, T1>& a, const Vec<3, T2>& b)
{
    return Vec<3, VecPrivate::common<T1, T2>>({
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    });
}

template<class T1, class T2>
inline VecPrivate::common<T1, T2> operator%(const Vec<2, T1>& a, const Vec<2, T2>& b)
{
    return a[0]*b[1] - a[1]*b[0];
}

template<class T>
inline Vec<2, T> rot_ccw(const Vec<2, T>& x) {
    return Vec<2, T>{ -x[1], x[0] };
}

template<class T>
inline Vec<2, T> rot_cw(const Vec<2, T>& x) {
    return Vec<2, T>{ x[1], -x[0] };
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator<<(const Vec<N, T>& a, unsigned int b)
{
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x << b; });
    return result;
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator>>(const Vec<N, T>& a, unsigned int b)
{
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x >> b; });
    return result;
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator&(const Vec<N, T>& a, unsigned int b)
{
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x & b; });
    return result;
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator&(const Vec<N, T>& a, const Vec<N, T>& b) {
    return elementwise(a, b, [](auto ai, auto bi) { return ai & bi; });
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator|(const Vec<N, T>& a, unsigned int b)
{
    Vec<N, T> result;
    std::transform(a.begin(), a.end(), result.begin(), [&](auto x) { return x | b; });
    return result;
}

template<unsigned int N, class T>
inline std::enable_if_t<std::is_integral<T>::value, Vec<N, T>>
operator|(const Vec<N, T>& a, const Vec<N, T>& b) {
    return elementwise(a, b, [](auto ai, auto bi) { return ai | bi; });
}

template<unsigned int N, class T>
inline T norm2(const Vec<N,T>& a) {
    return std::accumulate(a.begin(), a.end(), T(), [](const T& acc, const T& x) { return acc + x*x; });
}

template<unsigned int N, class T>
inline T norm1(const Vec<N,T>& a) {
    return std::accumulate(a.begin(), a.end(), T(), [](const T& acc, const T& x) { return acc + std::abs(x); });
}

template<unsigned int N, class T>
inline T normInf(const Vec<N,T>& a) {
    return std::accumulate(a.begin(), a.end(), T(), [](const T& acc, const T& x) { return acc + std::max(acc, std::abs(x)); });
}

template<unsigned int N, class T>
inline auto norm(const Vec<N,T>& a) -> decltype(auto) {
    return std::sqrt(norm2(a));
}

// Returns the index of coordinate with the maximum absolute value
template<unsigned int N, class T>
inline unsigned int primaryDim(const Vec<N, T>& x)
{
    return static_cast<unsigned int>(
        std::max_element(x.begin(), x.end(), [](const real_type& a, const real_type& b) {
            return std::abs(a) < std::abs(b);
        }) - x.begin());
}

// Returns integer in the range [0, 2*N-1] determining the primary orientation of vector x:
// 0: [1,0,0], 1: [-1,0,0], 2: [0,1,0], 3: [0,-1,0], ...
template<unsigned int N, class T>
inline unsigned int primaryOrientation(const Vec<N, T>& x)
{
    auto pd = primaryDim(x);
    return (pd<<1) | (x[pd] > 0? 1u: 0u);
}

} // s3dmm
