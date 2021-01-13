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

#include <initializer_list>
#include <tuple>
#include <algorithm>
#include <iterator>
#include <ostream>
#include <array>

#include <boost/assert.hpp>
#include <boost/static_assert.hpp>

namespace s3dmm {


template< unsigned int N_, class element_type_ = int >
class MultiIndex : public std::array<element_type_, N_>
{
public:
    static constexpr const unsigned int N = N_;
    using element_type = element_type_;
    using Base = std::array<element_type_, N_>;
    using This = MultiIndex<N, element_type>;
    using Base::begin;
    using Base::end;
    using Base::data;

    MultiIndex() {
        fill(element_type());
    }
    MultiIndex( std::initializer_list<element_type> l ) {
        BOOST_ASSERT(l.size() == N);
        std::copy(l.begin(), l.end(), data());
    }
    This& fill(const element_type& x) {
        std::fill(begin(), end(), x);
        return *this;
    }
    bool lt_all(const This& that) const {
        auto d1 = data();
        auto d2 = that.data();
        for (unsigned int i=0; i<N; ++i)
            if (!(d1[i] < d2[i]))
                return false;
        return true;
    }
    bool le_all(const This& that) const {
        auto d1 = data();
        auto d2 = that.data();
        for (unsigned int i=0; i<N; ++i)
            if (!(d1[i] <= d2[i]))
                return false;
        return true;
    }
    bool lt_some(const This& that) const {
        auto d1 = data();
        auto d2 = that.data();
        for (unsigned int i=0; i<N; ++i)
            if (d1[i] < d2[i])
                return true;
        return false;
    }
    bool le_some(const This& that) const {
        auto d1 = data();
        auto d2 = that.data();
        for (unsigned int i=0; i<N; ++i)
            if (d1[i] <= d2[i])
                return true;
        return false;
    }
    bool is_zero() const {
        return std::all_of(begin(), end(), [](auto& x) { return x == element_type(); });
    }
    template< unsigned int M >
    MultiIndex< M, element_type > extend() const {
        BOOST_STATIC_ASSERT(M >= N);
        MultiIndex< M, element_type > result;
        std::copy(begin(), end(), result.begin());
        return result;
    }
    template< unsigned int M >
    MultiIndex< M, element_type > extend(const element_type& x) const {
        BOOST_STATIC_ASSERT(M >= N);
        MultiIndex< M, element_type > result;
        std::copy(begin(), end(), result.begin());
        std::fill(result.begin()+N, result.end(), x);
        return result;
    }
    template< unsigned int M >
    MultiIndex< M, element_type > shrink() const {
        BOOST_STATIC_ASSERT(M <= N);
        MultiIndex< M, element_type > result;
        std::copy(begin(), begin()+M, result.begin());
        return result;
    }

    static This filled(const element_type& x) {
        return This().fill(x);
    }
    template<class T>
    MultiIndex<N,T> convertTo() const {
        MultiIndex<N,T> result;
        std::copy(begin(), begin()+N, result.begin());
        return result;
    }

    template<class T>
    This& operator+=(const MultiIndex<N,T>& that) {
        std::transform(begin(), end(), that.begin(), begin(), [](auto a, auto b) { return a + b; });
        return *this;
    }

    template<class T>
    This& operator-=(const MultiIndex<N,T>& that) {
        std::transform(begin(), end(), that.begin(), begin(), [](auto a, auto b) { return a - b; });
        return *this;
    }

    template<class T>
    This& operator+=(const T& x) {
        std::transform(begin(), end(), begin(), [&x](auto a) { return a + x; });
        return *this;
    }

    template<class T>
    This& operator-=(const T& x) {
        std::transform(begin(), end(), begin(), [&x](auto a) { return a - x; });
        return *this;
    }

    This& operator>>=(unsigned int x) {
        std::transform(begin(), end(), begin(), [&x](auto a) { return a >> x; });
        return *this;
    }

    This& operator<<=(unsigned int x) {
        std::transform(begin(), end(), begin(), [&x](auto a) { return a << x; });
        return *this;
    }

    template<class T>
    This& operator*=(const T& factor) {
        std::transform(begin(), end(), begin(), [&factor](auto a) { return a * factor; });
        return *this;
    }
};

template< class... Args >
inline auto multiIndex( Args ... args ) -> decltype(auto) {
    constexpr auto N = sizeof...(Args);
    using element_type = typename std::tuple_element<0, std::tuple<Args...>>::type;
    return MultiIndex<N, element_type>({args...});
    }

template< class element_type, class... Args >
inline auto multiIndexOfType( Args ... args ) -> decltype(auto) {
    constexpr auto N = sizeof...(Args);
    return MultiIndex<N, element_type>({args...});
    }

template< class C, unsigned int N, class T>
inline std::basic_ostream<C>& operator<<(std::basic_ostream<C>& s, const MultiIndex<N, T>& i)
{
    s << "[";
    if (N > 0) {
        copy(i.begin(), i.end()-1, std::ostream_iterator<T>(s, ","));
        s << i[N-1];
    }
    s << "]";
    return s;
}

} // s3dmm
