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

#include "BlockTreeFieldProvider.hpp"
#include "IndexTransform.hpp"
#include "HypercubeWalker.hpp"
#include "IncMultiIndex.hpp"
#include "DenseFieldInterpolatorTimers.hpp"

// #define S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
// #define S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP

// For some unknown reason, uncommenting this macro has the following effect (on Ubuntu 18.04):
// - when sparse field is read from SSD, the entire process speeds up several percent;
// - when sparse field is read from HDD, the entire process slows down several percent;
// When the macro is commented out, HDD outerpforms SSD, which looks like "latency hiding" applied
// to the reading of the sparse field.
// #define S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ

#if defined (S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE) || defined (S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP)
#include <iostream>
#include <boost/lexical_cast.hpp>
#define S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG
#endif // defined (S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE) || defined (S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP)

namespace s3dmm {

namespace detail {

template <unsigned int N>
struct DenseFieldInterpolatorStatePrinter {};

template <>
struct DenseFieldInterpolatorStatePrinter<1>
{
    static void printState(const std::vector<real_type>& denseField, unsigned int depth, real_type noFieldValue)
    {
        using namespace std;
        auto nodesPerEdge = IndexTransform<1>::verticesPerEdge<unsigned int>(depth);
        cout << "  ";
        for (auto i1=0u; i1<nodesPerEdge; ++i1) {
            auto ordinal = vertexIndexToOrdinal<2, unsigned int>({i1}, depth);
            auto& f = denseField[ordinal];
            char c = isnan(f)? '.': f == noFieldValue? 'o': '#';
            cout << c;
        }
        cout << endl;
    }
};

template <>
struct DenseFieldInterpolatorStatePrinter<2>
{
    static void printState(const std::vector<real_type>& denseField, unsigned int depth, real_type noFieldValue)
    {
        using namespace std;
        auto nodesPerEdge = IndexTransform<2>::verticesPerEdge<unsigned int>(depth);
        for (auto i2=0u; i2<nodesPerEdge; ++i2) {
            cout << "  ";
            for (auto i1=0u; i1<nodesPerEdge; ++i1) {
                auto ordinal = vertexIndexToOrdinal<2, unsigned int>({i1, i2}, depth);
                auto& f = denseField[ordinal];
                char c = isnan(f)? '.': f == noFieldValue? 'o': '#';
                cout << c;
            }
            cout << endl;
        }
    }
};

template <>
struct DenseFieldInterpolatorStatePrinter<3>
{
    static void printStateInLine(const std::vector<real_type>& denseField, unsigned int depth, real_type noFieldValue)
    {
        using namespace std;
        auto nodesPerEdge = IndexTransform<3>::verticesPerEdge<unsigned int>(depth);
        for (auto i2=0u; i2<nodesPerEdge; ++i2) {
            cout << "  ";
            for (auto i3=0u; i3<nodesPerEdge; ++i3) {
                for (auto i1=0u; i1<nodesPerEdge; ++i1) {
                    auto ordinal = vertexIndexToOrdinal<3, unsigned int>({i1, i2, i3}, depth);
                    auto& f = denseField[ordinal];
                    char c = isnan(f)? '.': f == noFieldValue? 'o': '#';
                    cout << c;
                }
                cout << "  ";
            }
            cout << endl;
        }
    }

    static void printStateMultiLine(const std::vector<real_type>& denseField, unsigned int depth, real_type noFieldValue)
    {
        using namespace std;
        auto nodesPerEdge = IndexTransform<3>::verticesPerEdge<unsigned int>(depth);
        for (auto i3=0u; i3<nodesPerEdge; ++i3) {
            cout << "  layer " << i3 << endl;
            for (auto i2=0u; i2<nodesPerEdge; ++i2) {
                cout << "  ";
                for (auto i1=0u; i1<nodesPerEdge; ++i1) {
                    auto ordinal = vertexIndexToOrdinal<3, unsigned int>({i1, i2, i3}, depth);
                    auto& f = denseField[ordinal];
                    char c = isnan(f)? '.': f == noFieldValue? 'o': '#';
                    cout << c;
                }
                cout << endl;
            }
            cout << endl;
        }
    }

    static void printState(const std::vector<real_type>& denseField, unsigned int depth, real_type noFieldValue) {
        printStateInLine(denseField, depth, noFieldValue);
    }
};

}

template<unsigned int N>
class DenseFieldInterpolator
{
public:
    using BlockId = typename BlockTree<N>::BlockId;
    using BlockIndex = typename BlockTree<N>::BlockIndex;
    using BT = typename Metadata<N>::BT;
    using dense_field_container_ref = std::vector<dfield_real>&;
    using Timers = DenseFieldInterpolatorTimers;

    std::shared_ptr<Timers> timers() const {
        return m_timers;
    }
    void setTimers(const std::shared_ptr<Timers>& timers) {
        m_timers = timers;
    }

    DenseFieldInterpolator(BlockTreeFieldProvider<N>& fp) : m_fp(fp)
    {}

    void interpolate(
            Vec2<real_type> fieldRange,
            dense_field_container_ref denseField,
            const BlockId& subtreeRoot)
    {
        // Obtain subtree and its nodes.
        ScopedTimerUser blockTreeNodesTimerUser(m_timers? &m_timers.get()->blockTreeNodesTimer: nullptr);
        auto& md = m_fp.metadata();
        auto subtreeNodes = md.blockTreeNodes(subtreeRoot); // TODO better: Copying!!!
        blockTreeNodesTimerUser.stop();

        interpolate(fieldRange, denseField, subtreeNodes);
    }

    void interpolate(
            Vec2<real_type>& fieldRange,
            dense_field_container_ref denseField,
            const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        // Obtain sparseField
        ScopedTimerUser sparseFieldTimerUser(m_timers? &m_timers.get()->sparseFieldTimer: nullptr);
        auto& sparseField = m_buf;
        m_fp.fieldValues(fieldRange, sparseField, subtreeNodes);
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ
        if (!sparseField.empty()) {
            volatile real_type x;
            x = sparseField.front();
            x = sparseField.back();
        }
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_ACCESS_SPARSE_FIELD_AFTER_READ
        sparseFieldTimerUser.stop();

        if (m_timers)
            ++m_timers.get()->invocationCount;

        // Interpolate dense field
        ScopedTimerUser interpolateTimerUser(m_timers? &m_timers.get()->interpolateTimer: nullptr);
        interpolate(denseField, sparseField, subtreeNodes);
    }

    static void interpolate(
            dense_field_container_ref denseField,
            const std::vector<real_type>& sparseField,
            const BlockTreeNodes<N, BT>& subtreeNodes)
    {
        using I = MultiIndex<N, unsigned int>;

        // Fill dense fields with sparse field values, where it is defined,
        // and with NANs where the sparse field is undefined.
        auto depth = subtreeNodes.maxDepth();
        auto nodesPerEdge = IndexTransform<N>::template verticesPerEdge<unsigned int>(depth);
        denseField.resize(IndexTransform<N>::vertexCount(depth));
        boost::range::fill(denseField, NAN);
        auto& n2i = subtreeNodes.data().n2i;
        foreach_byindex32(isparse, n2i) {
            auto idense = vertexIndexToOrdinal<N>(n2i[isparse], depth);
            denseField.at(idense) = make_dfield_real(sparseField.at(isparse));
        }

        // i-th component = how much the 1D dense index changes when
        // the i-th component of vector dense index increases by one
        BlockIndex denseStride;
        denseStride[0] = 1u;
        for (auto d=1; d<N; ++d)
            denseStride[d] = denseStride[d-1] * nodesPerEdge;

        const auto noFieldValue = make_dfield_real(BlockTreeFieldProvider<N>::noFieldValue());

        // deBUG, TODO: Remove
        using namespace std;
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
        cout << "******* Field interpolation *******" << endl;

        cout << "******* Initial state *******" << endl;
        auto printState = [&]() {
            detail::DenseFieldInterpolatorStatePrinter<N>::printState(denseField, depth, noFieldValue);
        };
#define PRINT_STATE printState();
#else // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
#define PRINT_STATE
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
        PRINT_STATE

        // Shift to apply to subtree level index to obtain dense index
        auto levelShift = depth;
        for (auto level=0u; level<depth; ++level, --levelShift) {
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG
            cout << "******* level " << level << " *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG
            // Pass all cubes of the current level. Interpolate dense field
            // in the middle of each edge, face, and cube, in the case the
            // sparse field is undefined.
            auto levelCubesPerEdge = 1u << level;
            BOOST_ASSERT(levelShift != 0u);
            auto levelDenseHalfStride = denseStride << (levelShift-1);
            auto vertexOffset = 1u << (levelShift-1);

            I levelCubeIndex;
            do {
                auto vertexIndexBase = levelCubeIndex << levelShift;

                auto interpolate1 = [&](const Hypercube<N,1>& hc) {
                    auto index = vertexIndexBase + (hc.origin << levelShift);
                    auto d = hc.dim[0];
                    index[d] += vertexOffset;
                    auto idenseMid = vertexIndexToOrdinal<N>(index, depth);
                    auto& midValue = denseField[idenseMid];
                    if (std::isnan(midValue)) {
                        // Field value in the middle of the edge is undefined
                        auto& startValue = denseField[idenseMid - levelDenseHalfStride[d]];
                        auto& endValue = denseField[idenseMid + levelDenseHalfStride[d]];
                        BOOST_ASSERT(!std::isnan(startValue) && !std::isnan(endValue));
                        if (startValue == noFieldValue || endValue == noFieldValue) {
                            midValue = noFieldValue;
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating NO FIELD at edge, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                        else {
                            midValue = make_real(0.5)*(startValue + endValue);
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating FIELD at edge, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                    }
//                    else if (midValue == noFieldValue) {
//                        auto& startValue = denseField[idenseMid - levelDenseHalfStride[d]];
//                        auto& endValue = denseField[idenseMid + levelDenseHalfStride[d]];
//                        BOOST_ASSERT(!std::isnan(startValue) && !std::isnan(endValue));
//                        if (!(startValue == noFieldValue || endValue == noFieldValue)) {
//                            midValue = make_real(0.5)*(startValue + endValue);
//#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                            cout << "******* Interpolating FIELD at edge, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
//#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                        }
//                    }
                };
                auto interpolate2 = [&](const Hypercube<N,2>& hc) {
                    auto index = vertexIndexBase + (hc.origin << levelShift);
                    auto d1 = hc.dim[0];
                    auto d2 = hc.dim[1];
                    index[d1] += vertexOffset;
                    index[d2] += vertexOffset;
                    auto idenseMid = vertexIndexToOrdinal<N>(index, depth);
                    auto& midValue = denseField[idenseMid];
                    if (std::isnan(midValue)) {
                        auto& startValue1 = denseField[idenseMid - levelDenseHalfStride[d1]];
                        auto& endValue1 = denseField[idenseMid + levelDenseHalfStride[d1]];
                        auto& startValue2 = denseField[idenseMid - levelDenseHalfStride[d2]];
                        auto& endValue2 = denseField[idenseMid + levelDenseHalfStride[d2]];
                        BOOST_ASSERT(!std::isnan(startValue1) && !std::isnan(endValue1) &&
                                     !std::isnan(startValue2) && !std::isnan(endValue2));
                        if (startValue1 == noFieldValue || endValue1 == noFieldValue ||
                            startValue2 == noFieldValue || endValue2 == noFieldValue) {
                            midValue = noFieldValue;
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating NO FIELD at face, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                        else {
                            midValue = make_real(0.25)*(
                                        startValue1 + endValue1 +
                                        startValue2 + endValue2);
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating FIELD at face, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                    }
//                    else if (midValue == noFieldValue) {
//                        auto& startValue1 = denseField[idenseMid - levelDenseHalfStride[d1]];
//                        auto& endValue1 = denseField[idenseMid + levelDenseHalfStride[d1]];
//                        auto& startValue2 = denseField[idenseMid - levelDenseHalfStride[d2]];
//                        auto& endValue2 = denseField[idenseMid + levelDenseHalfStride[d2]];
//                        BOOST_ASSERT(!std::isnan(startValue1) && !std::isnan(endValue1) &&
//                                     !std::isnan(startValue2) && !std::isnan(endValue2));
//                        if (!(startValue1 == noFieldValue || endValue1 == noFieldValue ||
//                            startValue2 == noFieldValue || endValue2 == noFieldValue))
//                        {
//                            midValue = make_real(0.25)*(
//                                        startValue1 + endValue1 +
//                                        startValue2 + endValue2);
//#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                            cout << "******* Interpolating FIELD at face, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
//#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                        }
//                    }
                };
                auto interpolate3 = [&](const Hypercube<N,3>& hc) {
                    auto index = vertexIndexBase + (hc.origin << levelShift);
                    auto d1 = hc.dim[0];
                    auto d2 = hc.dim[1];
                    auto d3 = hc.dim[2];
                    index[d1] += vertexOffset;
                    index[d2] += vertexOffset;
                    index[d3] += vertexOffset;
                    auto idenseMid = vertexIndexToOrdinal<N>(index, depth);
                    auto& midValue = denseField[idenseMid];
                    if (std::isnan(midValue)) {
                        auto& startValue1 = denseField[idenseMid - levelDenseHalfStride[d1]];
                        auto& endValue1 = denseField[idenseMid + levelDenseHalfStride[d1]];
                        auto& startValue2 = denseField[idenseMid - levelDenseHalfStride[d2]];
                        auto& endValue2 = denseField[idenseMid + levelDenseHalfStride[d2]];
                        auto& startValue3 = denseField[idenseMid - levelDenseHalfStride[d3]];
                        auto& endValue3 = denseField[idenseMid + levelDenseHalfStride[d3]];
                        BOOST_ASSERT(!std::isnan(startValue1) && !std::isnan(endValue1) &&
                                     !std::isnan(startValue2) && !std::isnan(endValue2) &&
                                     !std::isnan(startValue3) && !std::isnan(endValue3));
                        if (startValue1 == noFieldValue || endValue1 == noFieldValue ||
                            startValue2 == noFieldValue || endValue2 == noFieldValue ||
                            startValue3 == noFieldValue || endValue3 == noFieldValue) {
                            midValue = noFieldValue;
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating NO FIELD at volume, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                        else {
                            midValue = make_real(1./6)*(
                                        startValue1 + endValue1 +
                                        startValue2 + endValue2 +
                                        startValue3 + endValue3);
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                            cout << "******* Interpolating FIELD at volume, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
                        }
                    }
//                    else if (midValue == noFieldValue) {
//                        auto& startValue1 = denseField[idenseMid - levelDenseHalfStride[d1]];
//                        auto& endValue1 = denseField[idenseMid + levelDenseHalfStride[d1]];
//                        auto& startValue2 = denseField[idenseMid - levelDenseHalfStride[d2]];
//                        auto& endValue2 = denseField[idenseMid + levelDenseHalfStride[d2]];
//                        auto& startValue3 = denseField[idenseMid - levelDenseHalfStride[d3]];
//                        auto& endValue3 = denseField[idenseMid + levelDenseHalfStride[d3]];
//                        BOOST_ASSERT(!std::isnan(startValue1) && !std::isnan(endValue1) &&
//                                     !std::isnan(startValue2) && !std::isnan(endValue2) &&
//                                     !std::isnan(startValue3) && !std::isnan(endValue3));
//                        if (!(startValue1 == noFieldValue || endValue1 == noFieldValue ||
//                            startValue2 == noFieldValue || endValue2 == noFieldValue ||
//                            startValue3 == noFieldValue || endValue3 == noFieldValue))
//                        {
//                            midValue = make_real(1./6)*(
//                                        startValue1 + endValue1 +
//                                        startValue2 + endValue2 +
//                                        startValue3 + endValue3);
//#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                            cout << "******* Interpolating FIELD at volume, vertex " << index << " (" << idenseMid << ") *******" << endl;  // deBUG, TODO: Remove
//#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_OP
//                        }
//                    }
                };

                NestedHypercubeWalkHelper<N,N>::walk(interpolate3, interpolate2, interpolate1);
            }
            while (incMultiIndex(levelCubeIndex, levelCubesPerEdge));
#ifdef S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
            cout << "******* State at end of level " << level << " *******" << endl;
#endif // S3DMM_DEBUG_DENSE_FIELD_INTERPOLATOR_LOG_STATE
            PRINT_STATE
        }
        boost::range::transform(denseField, denseField.begin(), [&](const auto& x) {
            return x == noFieldValue? NAN: x;
        });
    }

private:
    BlockTreeFieldProvider<N>& m_fp;
    std::vector<real_type> m_buf;
    std::shared_ptr<Timers> m_timers;
};

} // s3dmm
