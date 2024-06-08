/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021-2024 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

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

#include "IndexRangeSplit.hpp"
#include "ScalarOrMultiIndex.hpp"

#include "s3dmm/IncMultiIndex.hpp"

#include <boost/assert.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <span>

namespace s3dmm
{

template <typename Blocks>
class BackToFrontOrder
{
public:
    using Block = typename Blocks::Block;
    using Splitter = typename Blocks::Splitter;

    static constexpr auto dim = Blocks::dim;

    using IndexVecTraits = ScalarOrMultiIndex<dim, unsigned int>;
    using CoordVecTraits = ScalarOrMultiIndex<dim, real_type>;

    using IndexVec = IndexVecTraits::type;
    using CoordVec = CoordVecTraits::type;

    class iterator:
        public boost::iterator_facade< iterator,
                                       const Block,
                                       std::forward_iterator_tag >
    {
    friend class BackToFrontOrder<Blocks>;
    public:
        const Block& dereference() const
        { return m_block = m_blocks->at(pos()); }

        void increment()
        {
            BOOST_ASSERT(!is_end());

            if (!inc_pos())
                next_macro_block();
        }

        bool equal(const iterator& that) const
        {
            if (is_end())
                return that.is_end();
            else if (that.is_end())
                return false;

            BOOST_ASSERT(m_blocks == that.m_blocks);
            BOOST_ASSERT(m_splits == that.m_splits);
            return m_pos == that.m_pos;
        }

    private:
        using MIndex = MultiIndex<dim, unsigned int>;

        iterator() = default;

        iterator(const Blocks* blocks,
                 const CoordVec& eye):
            m_blocks{ blocks },
            m_splits(
                [&]<size_t... I>(std::index_sequence<I...>)
                    -> std::array<IndexRangeSplit, dim>
                {
                    BOOST_ASSERT(m_blocks);
                    return { m_blocks->index_range_splitter(I)
                            (CoordVecTraits::element(eye, I))... };
                }
                (std::make_index_sequence<dim>()))
        {
            // Find first nonempty macro-block to start from
            next_macro_block();
        }

        bool is_valid_macro_index(unsigned int axis,
                                  unsigned int index) const noexcept
        {
            const auto& split = m_splits[axis];
            switch (index)
            {
            case 0: return !split.before.empty();
            case 1: return split.at != ~0u;
            case 2: return !split.after.empty();
            default:
                BOOST_ASSERT(false);
                __builtin_unreachable();
            }
        }

        bool is_valid_macro_pos(const MIndex& macro_pos) const noexcept
        {
            for (auto axis=0; axis<dim; ++axis)
                if (!is_valid_macro_index(axis, macro_pos[axis]))
                    return false;
            return true;
        }

        bool is_end() const noexcept
        { return m_blocks == nullptr; }

        void set_end() noexcept
        { m_blocks = nullptr; }

        void init_pos() noexcept
        {
            for (unsigned int axis=0; axis<dim; ++axis)
            {
                auto r = macro_range(axis);
                m_min[axis] = m_pos[axis] = r[0];
                m_max[axis] = r[1];
            }
        }

        IndexRange macro_range(unsigned int axis) const noexcept
        {
            IndexRange result;
            const auto& split = m_splits[axis];
            switch (m_macro_pos[axis])
            {
            case 0: return split.before;
            case 1: return { split.at, split.at + 1 };
            case 2: return split.after;
            default:
                BOOST_ASSERT(false);
                __builtin_unreachable();
            }
        }

        static std::span<const uint8_t> macro_order() noexcept
        {
            static_assert (dim > 0 && dim <= 3);

            if constexpr (dim == 1)
            {
                static constexpr uint8_t result[] = { 0,  2,  1 };
                return result;
            }
            else if constexpr (dim == 2)
            {
                static constexpr uint8_t result[] = {
                     0,  2,  6,  8, 1,  3,  5,  7,  4 };
                return result;
            }

            else if constexpr (dim == 3)
            {
                static constexpr uint8_t result[] = {
                     0,  2,  6,  8, 18, 20, 24, 26,
                     1,  3,  5,  7,  9, 11, 15, 17, 19, 21, 23, 25,
                     4, 10, 12, 14, 16, 22,
                    13 };
                return result;
            }

            __builtin_unreachable();
        }

        static MIndex macro_pos_from_ord(uint8_t ord) noexcept
        {
            static_assert (dim > 0 && dim <= 3);

            if constexpr (dim == 1)
                return { ord };

            else if constexpr (dim == 2)
                return { ord%3u, ord/3u };

            else if constexpr (dim == 3)
            {
                auto p0 = ord % 3u;
                ord /= 3u;
                return { p0, ord%3u, ord/3u };
            }

            __builtin_unreachable();
        }

        void next_macro_block()
        {
            auto order = macro_order();
            for (++m_macro_ord; m_macro_ord<order.size(); ++m_macro_ord)
            {
                m_macro_pos = macro_pos_from_ord(order[m_macro_ord]);
                if (is_valid_macro_pos(m_macro_pos))
                {
                    init_pos();
                    return;
                }
            }
            set_end();
        }

        bool inc_pos() noexcept
        {
            static_assert (dim > 0 && dim <= 3);

            if constexpr (dim == 1)
            {
                ++m_pos[0];
                return m_pos[0] < m_max[0];
            }
            else if constexpr (dim == 2)
            {
                auto s = m_pos[0] + m_pos[1];
                if (s + 2 == m_max[0] + m_max[1])
                    return false;
                if (m_pos[0] > m_min[0] && m_pos[1]+1 < m_max[1])
                {
                    --m_pos[0];
                    ++m_pos[1];
                }
                else if (s-m_min[1]+1 < m_max[0])
                    m_pos = { s-m_min[1]+1, m_min[1] };
                else
                    m_pos = { m_max[0]-1, s-m_max[0]+2 };
                return true;
            }
            else if constexpr (dim == 3)
            {
                auto s = m_pos[0] + m_pos[1] + m_pos[2];
                if (s + 3 == m_max[0] + m_max[1] + m_max[2])
                    return false;
                if (m_pos[0] > m_min[0] && m_pos[1]+1 < m_max[1])
                {
                    --m_pos[0];
                    ++m_pos[1];
                }
                else if (m_pos[0] + m_pos[1] > m_min[0] + m_min[1] &&
                         m_pos[2]+1 < m_max[2])
                {
                    if (s-m_min[1]-m_pos[2] < m_max[0]+1)
                        m_pos = { s-m_min[1]-m_pos[2]-1, m_min[1], m_pos[2]+1 };
                    else
                        m_pos = { m_max[0]-1, s-m_max[0]-m_pos[2], m_pos[2]+1 };
                }
                else
                {
                    ++s;
                    if (auto x0 = s-m_min[1]-m_min[2]; x0<m_max[0])
                        m_pos = { x0, m_min[1], m_min[2] };
                    else if (auto x1 = s-m_max[0]+1-m_min[2]; x1<m_max[1])
                        m_pos = { m_max[0]-1, x1, m_min[2] };
                    else
                    {
                        auto x2 = s+2-m_max[0]-m_max[1];
                        BOOST_ASSERT(x2 < m_max[2]);
                        m_pos = { m_max[0]-1, m_max[1]-1, x2 };
                    }
                }
                return true;
            }

            __builtin_unreachable();
        }

        IndexVec pos() const noexcept
        {
            MIndex result;
            for (unsigned int axis=0; axis<dim; ++axis)
                result[axis] = m_macro_pos[axis] == 2
                    ? m_max[axis] - 1 + m_min[axis] - m_pos[axis]
                    : m_pos[axis];
            return IndexVecTraits::fromMultiIndex(result);
        }

        const Blocks* m_blocks{};
        std::array<IndexRangeSplit, dim> m_splits;

        MIndex m_macro_pos;
        uint8_t m_macro_ord {0xff};

        MIndex m_pos;
        MIndex m_min;
        MIndex m_max;

        mutable Block m_block{};
    };

    class Range
    {
    public:
        Range(iterator begin, iterator end):
            m_begin{ std::move(begin) },
            m_end{ std::move(end) }
        {}

        iterator begin() const { return m_begin; }
        iterator end() const { return m_end; }

    private:
        iterator m_begin;
        iterator m_end;
    };

    explicit BackToFrontOrder(const Blocks& blocks):
        m_blocks{ blocks }
    {}

    explicit BackToFrontOrder(Blocks&&) = delete;

    Range range(const CoordVec& eye) const
    { return { {&m_blocks, eye }, {} }; }

private:
    const Blocks& m_blocks;
};

} // namespace s3dmm
