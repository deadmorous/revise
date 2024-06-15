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

#include <algorithm>
#include <span>
#include <vector>

namespace s3dmm
{

class NonUniformIndexRangeSplitter
{
public:
    NonUniformIndexRangeSplitter(unsigned int start_index,
                                 std::span<const real_type> split_coords,
                                 const CoordRange& coord_range):
        m_start_index{ start_index },
        m_coords( 2 + split_coords.size() )
    {
        BOOST_ASSERT(coord_range.is_normalized());
        m_coords.front() = coord_range[0];
        std::copy(split_coords.begin(), split_coords.end(), m_coords.begin()+1);
        m_coords.back() = coord_range[1];
        BOOST_ASSERT(std::is_sorted(m_coords.begin(), m_coords.end()));
    }

    IndexRangeSplit operator()(real_type x) const noexcept
    {
        auto it = std::lower_bound(m_coords.begin(),
                                   m_coords.end(),
                                   x);
        auto n = static_cast<unsigned int>( m_coords.size() - 1 );
        if (it == m_coords.begin())
            return { IndexRange::make_empty(),
                     ~0u,
                     IndexRange{m_start_index, m_start_index+n} };
        else if (it == m_coords.end())
            return { IndexRange{m_start_index, m_start_index+n},
                     ~0u,
                     IndexRange::make_empty() };
        auto at = m_start_index +
                  static_cast<unsigned int>(it - m_coords.begin()) - 1;
        return { {m_start_index, at}, at, {at+1, m_start_index+n} };
    }

    IndexRange index_range() const noexcept
    {
        return {
            m_start_index,
            m_start_index + static_cast<unsigned int>(m_coords.size()) - 1u
        };
    }

private:
    unsigned int m_start_index;
    std::vector<real_type> m_coords;
};

} // namespace s3dmm
