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

template<unsigned int N, class T>
struct ScalarOrMultiIndex
{
    using type = MultiIndex<N, T>;

    template<class F>
    static auto transform(const MultiIndex<N,T>& x, F f) -> decltype (auto)
    {
        MultiIndex<N, decltype (f(x[0]))> result;
        std::transform(x.begin(), x.end(), result.begin(), f);
        return result;
    }
    static MultiIndex<N,T> toMultiIndex(const MultiIndex<N,T>& x) {
        return x;
    }
    static MultiIndex<N,T> fromMultiIndex(const MultiIndex<N,T>& x) {
        return x;
    }
    static T& element(MultiIndex<N,T>& x, unsigned int index) {
        return x[index];
    }
    static const T& element(const MultiIndex<N,T>& x, unsigned int index) {
        return x[index];
    }
    static const T& min(const MultiIndex<N,T>& x) {
        return *std::min_element(x.begin(), x.end());
    }
    static const T& max(const MultiIndex<N,T>& x) {
        return *std::max_element(x.begin(), x.end());
    }
    template<class F>
    static void each(const MultiIndex<N,T>& x, F cb) {
        std::for_each(x.begin(), x.end(), cb);
    }
    template<class F>
    static void each(MultiIndex<N,T>& x, F cb) {
        std::for_each(x.begin(), x.end(), cb);
    }
    template<class F>
    static void each_indexed(const MultiIndex<N,T>& x, F cb) {
        for (auto i=0u; i<N; ++i)
            cb(x[i], i);
    }
    template<class F>
    static void each_indexed(MultiIndex<N,T>& x, F cb) {
        for (auto i=0u; i<N; ++i)
            cb(x[i], i);
    }
};
template<unsigned int N, class T>
using ScalarOrMultiIndex_t = typename ScalarOrMultiIndex<N, T>::type;

template<class T>
struct ScalarOrMultiIndex<1, T>
{
    using type = T;
    template<class F>
    static auto transform(const T& x, F f) -> decltype (auto)
    {
        return f(x);
    }
    static MultiIndex<1,T> toMultiIndex(const T& x) {
        return {x};
    }
    static T fromMultiIndex(const MultiIndex<1,T>& x) {
        return x[0];
    }
    static T& element(T& x, unsigned int index) {
        return x;
    }
    static const T& element(const T& x, unsigned int index) {
        return x;
    }
    static const T& min(const T& x) {
        return x;
    }
    static const T& max(const T& x) {
        return x;
    }
    template<class F>
    static void each(const T& x, F cb) {
        cb(x);
    }
    template<class F>
    static void each(T& x, F cb) {
        cb(x);
    }
    template<class F>
    static void each_indexed(const T& x, F cb) {
        cb(x, 0u);
    }
    template<class F>
    static void each_indexed(T& x, F cb) {
        cb(x, 0u);
    }
};

} // s3dmm
