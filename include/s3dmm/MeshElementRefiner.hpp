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

#include "BoundingBox.hpp"
#include "MeshElementType.hpp"
#include "HypercubeWalker.hpp"
#include "IndexTransform.hpp"
#include "IncMultiIndex.hpp"

#include <vector>
#include <boost/iterator/iterator_facade.hpp>

namespace s3dmm {

enum class MeshElementTopology {
    Simplex,
    Cube
};

namespace detail {

inline void computeLinearCombination(
    real_type *dst, unsigned int length,
    const real_type *vertexValues, unsigned int stride,
    unsigned int vertexCount, const real_type *weights)
{
    auto d = dst;
    for (auto icomponent=0u; icomponent<length; ++icomponent, ++d) {
        auto p = vertexValues + icomponent;
        *d = make_real(0);
        for (auto inode=0u; inode<vertexCount; ++inode, p+=stride) {
            *d += weights[inode]* (*p);
        }
    }
}

template <unsigned int N, MeshElementTopology Topology>
struct MeshElementHelper;

template<unsigned int N>
inline const MultiIndex<N, real_type>& vertexPos(const real_type *vertexValues, unsigned int stride, unsigned int index) {
    return *reinterpret_cast<const MultiIndex<N, real_type>*>(vertexValues + stride*index);
}

template <>
struct MeshElementHelper<1, MeshElementTopology::Cube>
{
    using vector_type = real_type;

    static constexpr unsigned int nodeCount() {
        return 2;
    }

    static const unsigned int *tecplotElementNodes() {
        static const unsigned int result[] = { 0, 1 };
        return result;
    }

    static MultiIndex<1, vector_type> elementBasis(const real_type *vertexValues, unsigned int stride) {
        return {
            vertexPos<1>(vertexValues, stride, 1) - vertexPos<1>(vertexValues, stride, 0)
        };
    }

    static void interpolate(
            const MultiIndex<1, real_type>& param,
            real_type *dst,
            unsigned int length,
            const real_type *vertexValues,
            unsigned int stride)
    {
        real_type weights[] = { param[0], make_real(1)-param[0] };
        computeLinearCombination(dst, length, vertexValues, stride, nodeCount(), weights);
    }
};

template <>
struct MeshElementHelper<2, MeshElementTopology::Cube>
{
    using vector_type = MultiIndex<2, real_type>;

    static constexpr unsigned int nodeCount() {
        return 4;
    }

    static const unsigned int *tecplotElementNodes() {
        static const unsigned int result[] = { 0, 1, 3, 2 };
        // static const unsigned int result[] = { 0, 1, 2, 3 };
        return result;
    }

    static MultiIndex<2, vector_type> elementBasis(const real_type *vertexValues, unsigned int stride) {
        return {
            vertexPos<2>(vertexValues, stride, 1) - vertexPos<2>(vertexValues, stride, 0),
            vertexPos<2>(vertexValues, stride, 3) - vertexPos<2>(vertexValues, stride, 0)
        };
    }

    static void interpolate(
            const MultiIndex<2, real_type>& param,
            real_type *dst,
            unsigned int length,
            const real_type *vertexValues,
            unsigned int stride)
    {
        auto& xi2 = param[0];
        auto xi1 = make_real(1) - xi2;
        auto& eta2 = param[1];
        auto eta1 = make_real(1) - eta2;

        real_type weights[] = { xi1*eta1, xi2*eta1, xi2*eta2, eta1*eta2 };
        computeLinearCombination(dst, length, vertexValues, stride, nodeCount(), weights);
    }
};

template <>
struct MeshElementHelper<3, MeshElementTopology::Cube>
{
    using vector_type = MultiIndex<3, real_type>;
    static MultiIndex<3, vector_type> elementBasis(const real_type *vertexValues, unsigned int stride) {
        return {
            vertexPos<3>(vertexValues, stride, 1) - vertexPos<3>(vertexValues, stride, 0),
            vertexPos<3>(vertexValues, stride, 3) - vertexPos<3>(vertexValues, stride, 0),
            vertexPos<3>(vertexValues, stride, 4) - vertexPos<3>(vertexValues, stride, 0)
        };
    }

    static constexpr unsigned int nodeCount() {
        return 8;
    }

    static const unsigned int *tecplotElementNodes() {
        static const unsigned int result[] = { 0, 1, 3, 2, 4, 5, 7, 6 };
        // static const unsigned int result[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        return result;
    }

    static void interpolate(
            const MultiIndex<3, real_type>& param,
            real_type *dst,
            unsigned int length,
            const real_type *vertexValues,
            unsigned int stride)
    {
        auto& xi2 = param[0];
        auto xi1 = make_real(1) - xi2;
        auto& eta2 = param[1];
        auto eta1 = make_real(1) - eta2;
        auto& zeta2 = param[2];
        auto zeta1 = make_real(1) - zeta2;
        real_type xieta[] = { xi1*eta1, xi2*eta1, xi2*eta2, eta1*eta2 };
        real_type weights[] = {
            xieta[0]*zeta1, xieta[1]*zeta1, xieta[2]*zeta1, xieta[3]*zeta1,
            xieta[0]*zeta2, xieta[1]*zeta2, xieta[2]*zeta2, xieta[3]*zeta2,
        };
        computeLinearCombination(dst, length, vertexValues, stride, nodeCount(), weights);
    }
};

template <>
struct MeshElementHelper<1, MeshElementTopology::Simplex> : MeshElementHelper<1, MeshElementTopology::Cube> {};

template <>
struct MeshElementHelper<2, MeshElementTopology::Simplex>
{
    using vector_type = MultiIndex<2, real_type>;

    static constexpr unsigned int nodeCount() {
        return 3;
    }

    static const unsigned int *tecplotElementNodes() {
        static const unsigned int result[] = { 0, 1, 2 };
        return result;
    }

    static MultiIndex<2, vector_type> elementBasis(const real_type *vertexValues, unsigned int stride) {
        return {
            vertexPos<2>(vertexValues, stride, 1) - vertexPos<2>(vertexValues, stride, 0),
            vertexPos<2>(vertexValues, stride, 2) - vertexPos<2>(vertexValues, stride, 0)
        };
    }

    static void interpolate(
            const MultiIndex<2, real_type>& /*param*/,
            real_type */*dst*/,
            unsigned int /*length*/,
            const real_type */*vertexValues*/,
            unsigned int /*stride*/)
    {
        // TODO
        throw std::runtime_error("MeshElementHelper<2, MeshElementTopology::Simplex>::interpolate() is not implemented");
    }
};

template <>
struct MeshElementHelper<3, MeshElementTopology::Simplex>
{
    using vector_type = MultiIndex<3, real_type>;
    static MultiIndex<3, vector_type> elementBasis(const real_type *vertexValues, unsigned int stride) {
        return {
            vertexPos<3>(vertexValues, stride, 1) - vertexPos<3>(vertexValues, stride, 0),
            vertexPos<3>(vertexValues, stride, 2) - vertexPos<3>(vertexValues, stride, 0),
            vertexPos<3>(vertexValues, stride, 3) - vertexPos<3>(vertexValues, stride, 0)
        };
    }

    static constexpr unsigned int nodeCount() {
        return 4;
    }

    static const unsigned int *tecplotElementNodes() {
        static const unsigned int result[] = { 0, 1, 2, 3 };
        return result;
    }

    static void interpolate(
            const MultiIndex<3, real_type>& /*param*/,
            real_type */*dst*/,
            unsigned int /*length*/,
            const real_type */*vertexValues*/,
            unsigned int /*stride*/)
    {
        // TODO
        throw std::runtime_error("MeshElementHelper<3, MeshElementTopology::Simplex>::interpolate() is not implemented");
    }
};

} // detail



template <unsigned int N, MeshElementTopology Topology>
class TriviallyRefinedMeshElement
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;

    explicit TriviallyRefinedMeshElement(real_type size) :
        m_size(size)
    {}

    real_type size() const {
        return m_size;
    }

    MultiIndex<N, unsigned int> maxIndex() const
    {
        MultiIndex<N, unsigned int> result;
        result.fill(1);
        return result;
    }

    static constexpr unsigned int count() {
        return detail::MeshElementHelper<N, Topology>::nodeCount();
    }

    void interpolate(real_type *dst, unsigned int length, const real_type *vertexValues, unsigned int stride) const
    {
        BOOST_ASSERT(length <= stride);
        for (auto i=0; i<count(); ++i, dst+=length, vertexValues+=stride)
            std::copy(vertexValues, vertexValues+length, dst);
    }

private:
    real_type m_size;
};

template <unsigned int N, MeshElementTopology Topology>
class MeshElementTrivialRefiner
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;

    MeshElementTrivialRefiner() = default;
    MeshElementTrivialRefiner(real_type) {}

    TriviallyRefinedMeshElement<N, Topology> refine(const real_type *vertexValues, unsigned int stride) const
    {
        BoundingBox<N, real_type> bb;
        auto d = vertexValues;
        for (auto i=0u, n=detail::MeshElementHelper<N, Topology>::nodeCount(); i<N; ++i, d+=stride)
            bb << *reinterpret_cast<const vector_type*>(d);
        auto size = ScalarOrMultiIndex<N, real_type>::max(bb.size());
        return TriviallyRefinedMeshElement<N, Topology>(size);
    }
};



template <unsigned int N, MeshElementTopology Topology>
class RefinedMeshElement
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;

    RefinedMeshElement(real_type size, const MultiIndex<N, unsigned int>& maxIndex) :
        m_size(size), m_maxIndex(maxIndex)
    {}

    real_type size() const {
        return m_size;
    }

    MultiIndex<N, unsigned int> maxIndex() const {
        return m_maxIndex;
    }

    unsigned int count() const {
        auto result = 1u;
        std::for_each(m_maxIndex.begin(), m_maxIndex.end(), [&result](auto x) {
            result *= x+1;
        });
        return result;
    }

    void interpolate(real_type *dst, unsigned int length, const real_type *vertexValues, unsigned int stride) const
    {
        // Copy values at all vertices to dst
        auto vertexCount = detail::MeshElementHelper<N, Topology>::nodeCount();
        auto vertices = detail::MeshElementHelper<N, Topology>::tecplotElementNodes();
        MultiIndex<N, unsigned int> dstIndex;
        for (auto ivertex=0u; ivertex<vertexCount; ++ivertex, inc01MultiIndex(dstIndex, m_maxIndex)) {
            auto src = vertexValues + vertices[ivertex]*stride;
            std::copy(src, src+length, dst+length*indexToOrdinal(dstIndex, m_maxIndex));
        }
        auto interpolate1_scaled = [&](const Hypercube<N,1>& hcScaled) {
            auto edgeDim = hcScaled.dim[0];
            auto maxEdgeIndex = m_maxIndex[edgeDim];
            if (maxEdgeIndex < 2)
                return;
            auto edgeStartIndex = hcScaled.origin;
            BOOST_ASSERT(edgeStartIndex[edgeDim] == 0);
            auto edgeDstDelta = length*indexToOrdinalStride(edgeDim, m_maxIndex);
            auto d1 = dst + length*indexToOrdinal(edgeStartIndex, m_maxIndex);
            auto d2 = d1 + edgeDstDelta*maxEdgeIndex;
            auto d = d1 + edgeDstDelta;
            auto h = make_real(1) / maxEdgeIndex;
            auto p = h;
            for (auto i=1u; i<maxEdgeIndex; ++i, p+=h, d+=edgeDstDelta)
                for (auto icomponent=0; icomponent<length; ++icomponent)
                    d[icomponent] = d1[icomponent] + p*(d2[icomponent]-d1[icomponent]);
        };
        auto interpolate1 = [&](const Hypercube<N,1>& hc) {
            interpolate1_scaled(scaledHypercube(hc, m_maxIndex));
        };
        auto interpolate2_scaled = [&](const Hypercube<N,2>& hcScaled) {
            auto faceDim1 = hcScaled.dim[0];
            auto faceDim2 = hcScaled.dim[1];
            auto maxFaceIndex1 = m_maxIndex[faceDim1];
            auto maxFaceIndex2 = m_maxIndex[faceDim2];
            if (maxFaceIndex1 < 2 || maxFaceIndex2 < 2)
                return;
            if (maxFaceIndex2 < maxFaceIndex1) {
                std::swap(faceDim1, faceDim2);
                std::swap(maxFaceIndex1, maxFaceIndex2);
            }
            Hypercube<N,1> hcEdgeScaled = {
                hcScaled.origin, {faceDim2}
            };
            auto& edgePos = hcEdgeScaled.origin[faceDim1];
            ++edgePos;
            for (auto i=1; i<maxFaceIndex1; ++i, ++edgePos)
                interpolate1_scaled(hcEdgeScaled);
        };
        auto interpolate2 = [&](const Hypercube<N,2>& hc) {
            interpolate2_scaled(scaledHypercube(hc, m_maxIndex));
        };
        auto interpolate3 = [&](const Hypercube<N,3>& hc) {
            auto dim = static_cast<unsigned int>(std::min_element(m_maxIndex.begin(), m_maxIndex.end()) - m_maxIndex.begin());
            auto maxIndex = m_maxIndex[dim];
            if (maxIndex < 2)
                return;
            auto faceDim1 = (dim + 1) % 3;
            auto faceDim2 = (dim + 2) % 3;
            Hypercube<N,2> hcFaceScaled = {
                hc.origin, {faceDim1, faceDim2}
            };
            auto& facePos = hcFaceScaled.origin[dim];
            for (auto i=1; i<maxIndex; ++i, ++facePos)
                interpolate2_scaled(hcFaceScaled);
        };
        NestedHypercubeWalkHelper<N,N>::walk(interpolate3, interpolate2, interpolate1);
    }

private:
    real_type m_size;
    MultiIndex<N, unsigned int> m_maxIndex;
};

template <unsigned int N, MeshElementTopology Topology>
class MeshElementRefiner
{
public:
    using vector_type = ScalarOrMultiIndex_t<N, real_type>;

    MeshElementRefiner() = default;
    explicit MeshElementRefiner(real_type param) : m_param(param)
    {
    }

    RefinedMeshElement<N, Topology> refine(const real_type *vertexValues, unsigned int stride) const
    {
        auto basis = detail::MeshElementHelper<N, Topology>::elementBasis(vertexValues, stride);
        MultiIndex<N, real_type> lengths;
        std::transform(basis.begin(), basis.end(), lengths.begin(), [](const vector_type& x) {
            real_type s = make_real(0);
            ScalarOrMultiIndex<N, real_type>::each(x, [&s](const real_type& x) {
                s += x*x;
            });
            return std::sqrt(s);
        });
        auto minLength = *std::min_element(lengths.begin(), lengths.end());
        auto maxLength = *std::max_element(lengths.begin(), lengths.end());
        auto size = minLength + m_param*(maxLength-minLength);
        MultiIndex<N, unsigned int> maxIndex;
        for (auto i=0u; i<N; ++i) {
            auto n = static_cast<unsigned int>(lengths[i]/size + make_real(0.5));
            if (n < 1)
                n = 1;
            basis[i] *= make_real(1)/n;
            maxIndex[i] = n;
        }
        return RefinedMeshElement<N, Topology>(size, maxIndex);
    }

private:
    real_type m_param = make_real(1);
};

template<MeshElementType Type> struct MeshElementRefinerDispatch;

template<> struct MeshElementRefinerDispatch<MeshElementType::Triangle>
{
    using Refiner = MeshElementTrivialRefiner<2, MeshElementTopology::Simplex>; // TODO
    using TrivialRefiner = MeshElementTrivialRefiner<2, MeshElementTopology::Simplex>;
};

template<> struct MeshElementRefinerDispatch<MeshElementType::Hexahedron>
{
    using Refiner = MeshElementRefiner<3, MeshElementTopology::Cube>;
    using TrivialRefiner = MeshElementTrivialRefiner<3, MeshElementTopology::Cube>;
};

template<> struct MeshElementRefinerDispatch<MeshElementType::Tetrahedron>
{
    using Refiner = MeshElementTrivialRefiner<3, MeshElementTopology::Simplex>; // TODO
    using TrivialRefiner = MeshElementTrivialRefiner<3, MeshElementTopology::Simplex>;
};

template<> struct MeshElementRefinerDispatch<MeshElementType::Quad>
{
    using Refiner = MeshElementRefiner<2, MeshElementTopology::Cube>;
    using TrivialRefiner = MeshElementTrivialRefiner<2, MeshElementTopology::Cube>;
};

class MeshElementRefinerParam
{
public:
    MeshElementRefinerParam() = default;
    MeshElementRefinerParam(real_type param) : m_param(param) {}
    template<MeshElementType Type> typename MeshElementRefinerDispatch<Type>::Refiner refiner() const {
        return typename MeshElementRefinerDispatch<Type>::Refiner(m_param);
    }
    template<MeshElementType Type> typename MeshElementRefinerDispatch<Type>::TrivialRefiner trivialRefiner() const {
        return typename MeshElementRefinerDispatch<Type>::TrivialRefiner();
    }
    bool needRefiner() const {
        return m_param < 1;
    }

private:
    real_type m_param = make_real(1);
};

} // s3dmm
