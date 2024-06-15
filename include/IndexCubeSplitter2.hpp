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
#include "NonUniformIndexRangeSplitter.hpp"

#include "s3dmm/IncMultiIndex.hpp"

#include <boost/assert.hpp>

#include <algorithm>
#include <array>

namespace s3dmm
{

template <unsigned int N>
class IndexCubeSplitter2 final
{
public:
    using IndexVecTraits = ScalarOrMultiIndex<N, unsigned int>;
    using Block = BoundingBox<N, unsigned int>;
    using IndexVec = IndexVecTraits::type;
    using Splitter = NonUniformIndexRangeSplitter;

    static constexpr auto dim = N;


    IndexCubeSplitter2(unsigned int level,
                       unsigned int split_count,
                       const BoundingBox<N, real_type>& bbox):
        m_level{level},
        m_split_count{split_count},
        m_bbox{bbox},
        m_axis_split{init_axis_split()}
    {}


    Block at(const IndexVec& pos) const noexcept
    {
        auto min = IndexVec{};
        auto max = IndexVec{};

        for (unsigned int axis=0; axis<dim; ++axis)
        {
            const auto& split_indices = m_axis_split[axis].split_indices;
            auto block_index = IndexVecTraits::element(pos, axis);
            IndexVecTraits::element(min, axis) = split_indices[block_index];
            IndexVecTraits::element(max, axis) = split_indices[block_index+1];
        }

        return Block{} << min << max;
    }

    IndexVec begin_index() const noexcept
    { return IndexVecTraits::fromMultiIndex( MIndex::filled(0) ); }

    IndexVec end_index() const noexcept
    {
        IndexVec result;
        for (unsigned int axis=0; axis<dim; ++axis)
            IndexVecTraits::element(result, axis) =
                m_axis_split[axis].splitter.index_range()[1];
        return result;
    }

    Splitter index_range_splitter(unsigned int axis) const noexcept
    { return m_axis_split[axis].splitter; }


    unsigned int level() const noexcept
    { return m_level; }

    unsigned int split_count() const noexcept
    { return m_split_count; }

    const BoundingBox<N, real_type>& bbox() const noexcept
    { return m_bbox; }

    auto all_blocks(bool add_trailing_empty) const
    {
        auto result = std::vector< Block >{};
        result.reserve(m_split_count);

        auto begin = begin_index();
        auto end = end_index();
        auto index = begin;

        do
            result.push_back(at(index));
        while (incMultiIndex(index, begin, end));

        if (add_trailing_empty)
            // Add empty boxes to match split count
            while (result.size() < m_split_count)
                result.push_back({});

        return result;
    }

private:
    using MIndex = MultiIndex<dim, unsigned int>;

    struct AxisSplit
    {
        std::vector<unsigned int> split_indices;
        NonUniformIndexRangeSplitter splitter;
    };

    using AxisSplitArr =
        std::array< AxisSplit, N >;

    AxisSplitArr init_axis_split()
    {
        BOOST_ASSERT(m_split_count > 0);
        auto size = 1u << m_level;
        auto max_split_count = 1u << (3u*m_level);
        unsigned int axis_split_counts[dim];
        auto split_count = std::min(max_split_count, m_split_count);

        // Note: The ` && axis<10` condition is redundant, but, perhaps due to
        // a compiler bug (g++ 11.4.0), the loop keeps running forever
        // without it.
        for (unsigned int axis=0; axis<dim && axis<10; ++axis)
        {
            if (split_count <= size)
            {
                axis_split_counts[axis] = split_count;
                split_count = 1;
            }
            else
            {
                axis_split_counts[axis] = size;
                split_count = std::max(1u, split_count / size);
            }
        }

        auto make_axis_split =
            [&](size_t axis) -> AxisSplit
        {
            auto split_count = axis_split_counts[axis];
            BOOST_ASSERT(split_count > 0);
            auto split_indices = std::vector<unsigned int>( split_count + 1 );
            split_indices.front() = 0;
            split_indices.back() = size;
            auto split_coords = std::vector<real_type>( split_count - 1 );
            auto coord_range = m_bbox.range(axis);
            for (unsigned int i=1; i<split_count; ++i)
            {
                auto split_index = i*size / split_count;
                split_indices[i] = split_index;
                split_coords[i-1] =
                    coord_range.origin() +
                    coord_range.length() * split_index / size;
            }
            return
                {
                    .split_indices = split_indices,
                    .splitter =
                        NonUniformIndexRangeSplitter{
                            0u,
                            split_coords,
                            m_bbox.range(axis) } };
        };

        return
            [&]<size_t... axis>(std::index_sequence<axis...>)
                -> AxisSplitArr
            { return { make_axis_split(axis) ... }; }
            (std::make_index_sequence<dim>());
    }

    unsigned int m_level;
    unsigned int m_split_count;
    BoundingBox<N, real_type> m_bbox;

    AxisSplitArr m_axis_split;
};

} // namespace s3dmm
