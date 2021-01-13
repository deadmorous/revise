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
#include <boost/core/ignore_unused.hpp>

namespace s3dmm {

namespace detail {

template<unsigned int N>
struct IndexTransformBase
{
    template<class Scalar>
    using Index = MultiIndex<N, Scalar>;

    template<class Scalar>
    static constexpr Scalar verticesPerEdge(unsigned int level) {
        return 1 + (1 << level);
    }

    static constexpr unsigned int vertexCount(unsigned int level)
    {
        auto result = 1u;
        for (auto d=0u; d<N; ++d)
            result *= verticesPerEdge<unsigned int>(level);
        return result;
    }

    static constexpr unsigned int blockCount(unsigned int level) {
        return 1 << (N*level);
    }

    template<class Scalar>
    static constexpr Scalar levelBlockBitMask(unsigned int level) {
        return (1<<(level+1))-1;
    }

    template<class Scalar>
    static bool isVertexIndexValid(const Index<Scalar>& index, unsigned int level)
    {
        auto vpe = verticesPerEdge<Scalar>(level);
        for (auto d=0u; d<N; ++d)
            if (index[d] >= vpe)
                return false;
        return true;
    }

    template<class Scalar>
    static bool isBlockIndexValid(const Index<Scalar>& index, unsigned int level) {
        return (index >> level) == Index<Scalar>();
    }

};

template<unsigned int N, unsigned int level, class Scalar>
struct FixedLevelIndexTransformBase : IndexTransformBase<N>
{
    using IndexTransformBase<N>::vertexCount;
    using IndexTransformBase<N>::verticesPerEdge;
    using IndexTransformBase<N>::blockCount;
    using IndexTransformBase<N>::levelBlockBitMask;
    static constexpr const Scalar VerticesPerEdge = IndexTransformBase<N>::template verticesPerEdge<Scalar>(level);
    static constexpr const unsigned int VertexCount = vertexCount(level);
    static constexpr const unsigned int BlockCount = blockCount();
    static constexpr const Scalar LevelBlockBitMask = IndexTransformBase<N>::template levelBlockBitMask<Scalar>();
    using Index = MultiIndex<N, Scalar>;
};

} // detail

template<unsigned int N>
struct IndexTransform : detail::IndexTransformBase<N>
{
    using Base = detail::IndexTransformBase<N>;
    template<class Scalar> using Index = typename Base::template Index<Scalar>;
    using Base::verticesPerEdge;
    using Base::vertexCount;
    using Base::blockCount;
    using Base::levelBlockBitMask;

    template<class Scalar>
    static unsigned int blockIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT((index >> level) == Index<Scalar>());
        auto result = 0u;
        for (auto d=N-1; d!=~0u; --d)
            result = (result << level) | index[d];
        return result;
    }

    template<class Scalar>
    static Index<Scalar> blockOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < blockCount(level));
        Index<Scalar> result;
        auto mask = levelBlockBitMask(level);
        for (auto d=0; d<N; ++d, ordinal>>=level)
            result[d] = static_cast<Scalar>(ordinal & mask);
        return result;
    }

    template<class Scalar>
    static unsigned int vertexIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        auto result = 0u;
        for (auto d=N-1; d!=~0u; --d)
            result = ((result << level)+result) + index[d];
        return result;
    }

    template<class Scalar>
    static Index<Scalar> vertexOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < blockCount(level));
        Index<Scalar> result;
        auto vpe = verticesPerEdge(level);
        for (auto d=0; d<N; ++d, ordinal/=vpe)
            result[d] = ordinal % vpe;
        return result;
    }
};

template<>
struct IndexTransform<1> : detail::IndexTransformBase<1>
{
    using Base = detail::IndexTransformBase<1>;
    template<class Scalar> using Index = typename Base::Index<Scalar>;
    using Base::vertexCount;
    using Base::blockCount;
    using Base::levelBlockBitMask;

    template<class Scalar>
    static unsigned int blockIndexToOrdinal(const Index<Scalar>& index, unsigned int level = 0)
    {
        BOOST_ASSERT((index >> level) == Index<Scalar>());
        boost::ignore_unused(level);
        return index[0];
    }

    template<class Scalar>
    static Index<Scalar> blockOrdinalToIndex(unsigned int ordinal, unsigned int level = 0)
    {
        BOOST_ASSERT(ordinal < blockCount(level));
        boost::ignore_unused(level);
        return {ordinal};
    }

    template<class Scalar>
    static unsigned int vertexIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        boost::ignore_unused(level);
        return index[0];
    }

    template<class Scalar>
    static Index<Scalar> vertexOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < vertexCount(level));
        boost::ignore_unused(level);
        return {ordinal};
    }
};

template<>
struct IndexTransform<2> : detail::IndexTransformBase<2>
{
    using Base = detail::IndexTransformBase<2>;
    template<class Scalar> using Index = typename Base::Index<Scalar>;
    using Base::vertexCount;
    using Base::blockCount;
    using Base::levelBlockBitMask;

    template<class Scalar>
    static unsigned int blockIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT((index >> level) == Index<Scalar>());
        return index[0] | (index[1]<<level);
    }

    template<class Scalar>
    static Index<Scalar> blockOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < blockCount(level));
        auto mask = levelBlockBitMask<Scalar>(level);
        return {ordinal&mask, ordinal>>level};
    }

    template<class Scalar>
    static unsigned int vertexIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        return index[0] + (index[1]<<level) + index[1];
    }

    template<class Scalar>
    static Index<Scalar> vertexOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < vertexCount(level));
        auto vpe = verticesPerEdge<Scalar>(level);
        return {ordinal%vpe, ordinal/vpe};
    }
};

template<>
struct IndexTransform<3> : detail::IndexTransformBase<3>
{
    using Base = detail::IndexTransformBase<3>;
    template<class Scalar> using Index = typename Base::Index<Scalar>;
    using Base::vertexCount;
    using Base::blockCount;
    using Base::levelBlockBitMask;

    template<class Scalar>
    static unsigned int blockIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT((index >> level) == Index<Scalar>());
        return index[0] | (index[1]<<level) | (index[2]<<(level<<1));
    }

    template<class Scalar>
    static Index<Scalar> blockOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < blockCount(level));
        auto mask = levelBlockBitMask<Scalar>(level);
        return {ordinal&mask, (ordinal>>level)&mask, (ordinal>>(level<<1))};
    }

    template<class Scalar>
    static unsigned int vertexIndexToOrdinal(const Index<Scalar>& index, unsigned int level)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        auto result = (index[2] << level) + index[2] + index[1];
        result = (result << level) + result + index[0];
        return result;
    }

    template<class Scalar>
    static Index<Scalar> vertexOrdinalToIndex(unsigned int ordinal, unsigned int level)
    {
        BOOST_ASSERT(ordinal < vertexCount(level));
        auto vpe = verticesPerEdge<Scalar>(level);
        auto o2 = ordinal / vpe;
        return {ordinal%vpe, o2%vpe, o2/vpe};
    }
};



template<unsigned int N, unsigned int level, class Scalar>
struct FixedLevelIndexTransform : detail::FixedLevelIndexTransformBase<N, level, Scalar>
{
    using Base = detail::FixedLevelIndexTransformBase<N, level, Scalar>;
    using typename Base::Index;
    using Base::VerticesPerEdge;
    using Base::LevelBlockBitMask;
    using Base::VertexCount;
    using Base::BlockCount;
    static unsigned int blockIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT((index >> level) == Index());
        auto result = 0u;
        for (auto d=N-1; d!=~0u; --d)
            result = (result << level) | index[d];
        return result;
    }

    static Index blockOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < BlockCount);
        Index result;
        for (auto d=0; d<N; ++d, ordinal>>=level)
            result[d] = ordinal & LevelBlockBitMask;
        return result;
    }

    static unsigned int vertexIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        auto result = 0u;
        for (auto d=N-1; d!=~0u; --d)
            result = ((result << level)+result) + index[d];
        return result;
    }

    static Index vertexOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < BlockCount);
        Index result;
        auto vpe = VerticesPerEdge;
        for (auto d=0; d<N; ++d, ordinal/=vpe)
            result[d] = ordinal % vpe;
        return result;
    }
};

template<unsigned int level, class Scalar>
struct FixedLevelIndexTransform<1, level, Scalar> : detail::FixedLevelIndexTransformBase<1, level, Scalar>
{
    using Base = detail::FixedLevelIndexTransformBase<1, level, Scalar>;
    using typename Base::Index;
    using Base::VerticesPerEdge;
    using Base::LevelBlockBitMask;
    using Base::VertexCount;
    using Base::BlockCount;

    static unsigned int blockIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT((index >> level) == Index());
        return index[0];
    }

    static Index blockOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < BlockCount);
        return {ordinal};
    }

    static unsigned int vertexIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        return index[0];
    }

    static Index vertexOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < VertexCount);
        return {ordinal};
    }
};

template<unsigned int level, class Scalar>
struct FixedLevelIndexTransform<2, level, Scalar> : detail::FixedLevelIndexTransformBase<2, level, Scalar>
{
    using Base = detail::FixedLevelIndexTransformBase<2, level, Scalar>;
    using typename Base::Index;
    using Base::VerticesPerEdge;
    using Base::LevelBlockBitMask;
    using Base::VertexCount;
    using Base::BlockCount;
    static unsigned int blockIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT((index >> level) == Index());
        return index[0] | (index[1]<<level);
    }

    static Index blockOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < BlockCount);
        return {ordinal&LevelBlockBitMask, ordinal>>level};
    }

    static unsigned int vertexIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        return index[0] + (index[1]<<level) + index[1];
    }

    static Index vertexOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < VertexCount);
        auto vpe = VerticesPerEdge;
        return {ordinal%vpe, ordinal/vpe};
    }
};

template<unsigned int level, class Scalar>
struct FixedLevelIndexTransform<3, level, Scalar> : detail::FixedLevelIndexTransformBase<3, level, Scalar>
{
    using Base = detail::FixedLevelIndexTransformBase<3, level, Scalar>;
    using Base::VerticesPerEdge;
    using typename Base::Index;
    using Base::LevelBlockBitMask;
    using Base::VertexCount;
    using Base::BlockCount;
    static unsigned int blockIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT((index >> level) == Index());
        return index[0] | (index[1]<<level) | (index[2]<<(level<<1));
    }

    static Index blockOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < BlockCount);
        return {ordinal&LevelBlockBitMask, (ordinal>>level)&LevelBlockBitMask, (ordinal>>(level<<1))};
    }

    static unsigned int vertexIndexToOrdinal(const Index& index)
    {
        BOOST_ASSERT(isVertexIndexValid(index, level));
        auto result = (index[2] << level) + index[2] + index[1];
        result = (result << level) + result + index[0];
        return result;
    }

    static Index vertexOrdinalToIndex(unsigned int ordinal)
    {
        BOOST_ASSERT(ordinal < VertexCount);
        auto vpe = VerticesPerEdge;
        auto o2 = ordinal / vpe;
        return {ordinal%vpe, o2%vpe, o2/vpe};
    }
};


template<unsigned int N, class Scalar>
inline unsigned int blockIndexToOrdinal(const MultiIndex<N, Scalar>& index, unsigned int level)
{
    return IndexTransform<N>::blockIndexToOrdinal(index, level);
}

template<unsigned int N, class Scalar>
inline MultiIndex<N, Scalar> blockOrdinalToIndex(unsigned int ordinal, unsigned int level) {
    return IndexTransform<N>::template blockOrdinalToIndex<Scalar>(ordinal, level);
}

template<unsigned int N, unsigned int level, class Scalar>
inline unsigned int blockIndexToOrdinal(const MultiIndex<N, Scalar>& index)
{
    return FixedLevelIndexTransform<N, level, Scalar>::blockIndexToOrdinal(index);
}

template<unsigned int N, unsigned int level, class Scalar>
inline MultiIndex<N, Scalar> blockOrdinalToIndex(unsigned int ordinal) {
    return FixedLevelIndexTransform<N, level, Scalar>::blockOrdinalToIndex(ordinal);
}




template<unsigned int N, class Scalar>
inline unsigned int vertexIndexToOrdinal(const MultiIndex<N, Scalar>& index, unsigned int level)
{
    return IndexTransform<N>::vertexIndexToOrdinal(index, level);
}

template<unsigned int N, class Scalar>
inline MultiIndex<N, Scalar> vertexOrdinalToIndex(unsigned int ordinal, unsigned int level) {
    return IndexTransform<N>::template vertexOrdinalToIndex<Scalar>(ordinal, level);
}

template<unsigned int N, unsigned int level, class Scalar>
inline unsigned int vertexIndexToOrdinal(const MultiIndex<N, Scalar>& index)
{
    return FixedLevelIndexTransform<N, level, Scalar>::vertexIndexToOrdinal(index);
}

template<unsigned int N, unsigned int level, class Scalar>
inline MultiIndex<N, Scalar> vertexOrdinalToIndex(unsigned int ordinal) {
    return FixedLevelIndexTransform<N, level, Scalar>::vertexOrdinalToIndex(ordinal);
}



namespace detail {

template<unsigned int N, class Scalar> struct IndexToOrdinalHelper {};

template<class Scalar>
struct IndexToOrdinalHelper<1, Scalar> {
    static unsigned int f(const MultiIndex<1, Scalar>& index, const MultiIndex<1, Scalar>& maxIndex)
    {
        BOOST_ASSERT(index[0] <= maxIndex[0]);
        boost::ignore_unused(maxIndex);
        return static_cast<unsigned int>(index[0]);
    }
};

template<class Scalar>
struct IndexToOrdinalHelper<2, Scalar> {
    static unsigned int f(const MultiIndex<2, Scalar>& index, const MultiIndex<2, Scalar>& maxIndex)
    {
        BOOST_ASSERT(index[0] <= maxIndex[0]);
        BOOST_ASSERT(index[1] <= maxIndex[1]);
        return static_cast<unsigned int>(index[0]) +
                (static_cast<unsigned int>(maxIndex[0])+1)*index[1];
    }
};

template<class Scalar>
struct IndexToOrdinalHelper<3, Scalar> {
    static unsigned int f(const MultiIndex<3, Scalar>& index, const MultiIndex<3, Scalar>& maxIndex)
    {
        BOOST_ASSERT(index[0] <= maxIndex[0]);
        BOOST_ASSERT(index[1] <= maxIndex[1]);
        BOOST_ASSERT(index[2] <= maxIndex[2]);
        return static_cast<unsigned int>(index[0]) +
                (static_cast<unsigned int>(maxIndex[0])+1) *
                (index[1] + (static_cast<unsigned int>(maxIndex[1])+1)*index[2]);
    }
};

} // detail

template<unsigned int N, class Scalar>
inline unsigned int indexToOrdinal(const MultiIndex<N, Scalar>& index, const MultiIndex<N, Scalar>& maxIndex) {
    return detail::IndexToOrdinalHelper<N, Scalar>::f(index, maxIndex);
}

//template<unsigned int N, class Scalar>
//inline unsigned int indexToOrdinal(const MultiIndex<N, Scalar>& index, const MultiIndex<N, Scalar>& maxIndex) {
//{
//    auto result = static_cast<unsigned int>(index[N-1]);
//    for (auto i=2u; i<=N; ++i)
//        result = result*(maxIndex[N-i]+1) + index[N-i];
//    return result;
//}


namespace detail {

template<unsigned int N, class Scalar> struct IndexToOrdinalStrideHelper {};

template<class Scalar>
struct IndexToOrdinalStrideHelper<1, Scalar> {
    static unsigned int f(unsigned int dim, const MultiIndex<1, Scalar>& maxIndex)
    {
        BOOST_ASSERT(dim < 1);
        boost::ignore_unused(dim, maxIndex);
        return 1u;
    }
};

template<class Scalar>
struct IndexToOrdinalStrideHelper<2, Scalar> {
    static unsigned int f(unsigned int dim, const MultiIndex<2, Scalar>& maxIndex)
    {
        BOOST_ASSERT(dim < 2);
        return dim? maxIndex[0] + 1u: 1u;
    }
};

template<class Scalar>
struct IndexToOrdinalStrideHelper<3, Scalar> {
    static unsigned int f(unsigned int dim, const MultiIndex<3, Scalar>& maxIndex)
    {
        BOOST_ASSERT(dim < 3);
        return dim == 0?
                    1u: dim == 1?
                        static_cast<unsigned int>(maxIndex[0]) + 1u:
                        (static_cast<unsigned int>(maxIndex[0]) + 1u)*(maxIndex[1] + 1u);
    }
};

} // detail

template<unsigned int N, class Scalar>
inline unsigned int indexToOrdinalStride(unsigned int dim, const MultiIndex<N, Scalar>& maxIndex)
{
    return detail::IndexToOrdinalStrideHelper<N, Scalar>::f(dim, maxIndex);
}

//template<unsigned int N, class Scalar>
//inline unsigned int indexToOrdinalStride(unsigned int dim, const MultiIndex<N, Scalar>& maxIndex)
//{
//    BOOST_ASSERT(dim < N);
//    auto result = 1u;
//    for (auto i=1u; i<dim; ++i)
//        result *= maxIndex[i] + 1;
//    return result;
//}

} // s3dmm
