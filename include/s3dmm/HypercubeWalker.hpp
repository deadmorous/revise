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

#include "Vec.hpp"
#include <boost/assert.hpp>

namespace s3dmm {

template <unsigned int N, unsigned int D>
struct Hypercube
{
    MultiIndex<N, unsigned int> origin;
    MultiIndex<D, unsigned int> dim;
};

template <unsigned int N, unsigned int D>
inline Hypercube<N,D> scaledHypercube(const Hypercube<N,D>& hc, const MultiIndex<N, unsigned int>& s)
{
    auto result = hc;
    for (auto i=0; i<N; ++i) {
        BOOST_ASSERT(hc.origin[i] <= 1);
        if (hc.origin[i] == 1)
            result.origin[i] = s[i];
    }
    return result;
}

template <unsigned int N, unsigned int D>
struct HypercubeWalker {};

template <>
struct HypercubeWalker<1,1> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<1,1>{{0}, {0}});
    }
};
template <>
struct HypercubeWalker<2,1> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<2,1>{{0,0}, {0}});
        cb(Hypercube<2,1>{{0,0}, {1}});
        cb(Hypercube<2,1>{{0,1}, {0}});
        cb(Hypercube<2,1>{{1,0}, {1}});
    }
};
template <>
struct HypercubeWalker<2,2> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<2,2>{{0,0}, {0,1}});
    }
};
template <>
struct HypercubeWalker<3,1> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<3,1>{{0,0,0}, {0}});
        cb(Hypercube<3,1>{{0,0,0}, {1}});
        cb(Hypercube<3,1>{{0,0,0}, {2}});
        cb(Hypercube<3,1>{{0,0,1}, {0}});
        cb(Hypercube<3,1>{{0,0,1}, {1}});
        cb(Hypercube<3,1>{{0,1,0}, {0}});
        cb(Hypercube<3,1>{{0,1,0}, {2}});
        cb(Hypercube<3,1>{{0,1,1}, {0}});
        cb(Hypercube<3,1>{{1,0,0}, {1}});
        cb(Hypercube<3,1>{{1,0,0}, {2}});
        cb(Hypercube<3,1>{{1,0,1}, {1}});
        cb(Hypercube<3,1>{{1,1,0}, {2}});
    }
};
template <>
struct HypercubeWalker<3,2> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<3,2>{{0,0,0}, {0,1}});
        cb(Hypercube<3,2>{{0,0,0}, {0,2}});
        cb(Hypercube<3,2>{{0,0,0}, {1,2}});
        cb(Hypercube<3,2>{{0,0,1}, {0,1}});
        cb(Hypercube<3,2>{{0,1,0}, {0,2}});
        cb(Hypercube<3,2>{{1,0,0}, {1,2}});
    }
};
template <>
struct HypercubeWalker<3,3> {
    template<class F> static void walk(F cb) {
        cb(Hypercube<3,3>{{0,0,0}, {0,1,2}});
    }
};

template<unsigned int N, unsigned int D>
struct NestedHypercubeWalker
{
    template <class FD, class ... FOther>
    static void walk(FD cb, FOther ... otherCb) {
        NestedHypercubeWalker<N,D-1>::walk(otherCb...);
        HypercubeWalker<N,D>::walk(cb);
    }
};

template<unsigned int N>
struct NestedHypercubeWalker<N, 1>
{
    template <class FD>
    static void walk(FD cb) {
        HypercubeWalker<N,1>::walk(cb);
    }
};

template<unsigned int N, unsigned int D, class FD, class ... FOther>
inline void walkNestedHypercubes(FD cb, FOther ... cbOther) {
    NestedHypercubeWalker<N,D>::walk(cb, cbOther...);
}

template<unsigned int N, unsigned int D>
struct NestedHypercubeWalkHelper {};

template<unsigned int N>
struct NestedHypercubeWalkHelper<N, 3>
{
    template<class F3, class F2, class F1>
    static void walk(F3 cb3, F2 cb2, F1 cb1) {
        walkNestedHypercubes<N,3>(cb3, cb2, cb1);
    }
};

template<unsigned int N>
struct NestedHypercubeWalkHelper<N, 2>
{
    template<class F3, class F2, class F1>
    static void walk(F3 /*cb3*/, F2 cb2, F1 cb1) {
        walkNestedHypercubes<N,2>(cb2, cb1);
    }
};

template<unsigned int N>
struct NestedHypercubeWalkHelper<N, 1>
{
    template<class F3, class F2, class F1>
    static void walk(F3 /*cb3*/, F2 /*cb2*/, F1 cb1) {
        walkNestedHypercubes<N,1>(cb1);
    }
};

} // s3dmm
