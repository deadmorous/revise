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
#include "TreeBlockId.hpp"
#include "BoundingCube.hpp"

#include <boost/range/algorithm/max_element.hpp>

namespace s3dmm {

template <unsigned int N>
class ImageToBlockIndexTransform
{
public:
    using Index = MultiIndex<N, unsigned int>;
    using BlockId = detail::TreeBlockId<N>;
    using vector_type = typename BoundingBox<N, real_type>::vector_type;

    ImageToBlockIndexTransform(const Index& imageSize, unsigned int depth) :
        m_sufficientDepth(sufficientTreeDepth(imageSize)),
        m_scaleFactor(0)
    {
        // Compute m_indexOffset and m_indexShift
        auto sufficientVoxelsPerEdge = 1u << m_sufficientDepth;
        m_indexOffset = (Index::filled(sufficientVoxelsPerEdge) - imageSize) >> 1;
        if (m_sufficientDepth > depth)
            m_indexShift = m_sufficientDepth - depth;
        else {
            BOOST_ASSERT(m_sufficientDepth == depth);
            m_indexShift = 0;
        }
    }

    ImageToBlockIndexTransform(
            const Index& imageSize,
            const BlockId& blockId,
            unsigned int blockDepth,
            const BoundingCube<N, real_type>& blockPos) :
        m_sufficientDepth(sufficientTreeDepth(imageSize))
    {
        auto sufficientVoxelsPerEdge = 1u << m_sufficientDepth;
        auto depth = blockId.level + blockDepth;

        // Compute m_indexOffset and m_indexShift
        m_indexOffset = (Index::filled(sufficientVoxelsPerEdge) - imageSize) >> 1;
        if (m_sufficientDepth > depth)
            m_indexShift = m_sufficientDepth - depth;
        else {
            BOOST_ASSERT(m_sufficientDepth == depth);
            m_indexShift = 0;
        }
        m_indexShift += blockId.level;
        m_indexOffset -= blockId.location << (m_sufficientDepth-blockId.level);

        // Compute m_scaleFactor and m_blockOrigin
        m_scaleFactor = blockPos.size() / (sufficientVoxelsPerEdge >> blockId.level);
        m_blockOrigin = blockPos.min();
    }

    Index operator()(const Index& imageIndex) const {
        return (imageIndex + m_indexOffset) >> m_indexShift;
    }

    vector_type pos(const Index& imageIndex) const
    {
        auto index = imageIndex + m_indexOffset;
        static const auto _05 = MultiIndex<N, real_type>::filled(make_real(0.5));
        auto scaled = (index.template convertTo<real_type>() + _05)*m_scaleFactor;
        return m_blockOrigin + ScalarOrMultiIndex<N, real_type>::fromMultiIndex(scaled);
    }

    static unsigned int sufficientTreeDepth(const Index& imageSize)
    {
        auto maxImageSize = *boost::range::max_element(imageSize);
        auto sufficientVoxelsPerEdge = 1u;
        auto result = 0u;
        while (sufficientVoxelsPerEdge < maxImageSize) {
            sufficientVoxelsPerEdge <<= 1;
            ++result;
        }
        return result;
    }

    static std::pair<Index, Index> imageIndexRangeForBlock(
        const Index& imageSize, const BlockId& blockId)
    {
        auto sufficientDepth = sufficientTreeDepth(imageSize);
        auto sufficientVoxelsPerEdge = 1u << sufficientDepth;
        auto indexOffset = (Index::filled(sufficientVoxelsPerEdge) - imageSize) >> 1;
        auto blockLocationShift = sufficientDepth - blockId.level;
        auto i1 = blockId.location << blockLocationShift;
        auto i2 = (blockId.location+Index::filled(1)) << blockLocationShift;
        auto i1i = indexOffset;
        auto i2i = indexOffset + imageSize;
        if (i2.le_some(i1i))
            return { i1i, i1i };
        if (i2i.le_some(i1))
            return { i2i, i2i };
        for (auto d=0; d<N; ++d) {
            if (i1[d] < i1i[d])
                i1[d] = i1i[d];
            if (i2[d] > i2i[d])
                i2[d] = i2i[d];
        }
        return {i1-indexOffset, i2-indexOffset};
    }

private:
    unsigned int m_sufficientDepth;
    Index m_indexOffset;
    unsigned int m_indexShift;
    vector_type m_blockOrigin;
    real_type m_scaleFactor {};
};

} // s3dmm
