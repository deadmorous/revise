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

namespace s3dmm
{

class UniformIndexRangeSplitter
{
public:
    UniformIndexRangeSplitter(const IndexRange& index_range,
                              const CoordRange& coord_range):
        m_index_range{ index_range },
        m_coord_range{ coord_range },
        m_factor{ (m_index_range[1] - m_index_range[0]) /
                 (m_coord_range[1] - m_coord_range[0]) }
    {
        BOOST_ASSERT(m_coord_range.is_normalized());
        BOOST_ASSERT(m_index_range.is_normalized());
    }

    IndexRangeSplit operator()(real_type x) const noexcept
    {
        if (x <= m_coord_range[0])
            return { IndexRange::make_empty(), ~0u, m_index_range };
        else if (x >= m_coord_range[1])
            return { m_index_range, ~0u, IndexRange::make_empty() };
        auto at = static_cast<unsigned int>(
                      (x - m_coord_range[0]) * m_factor) + m_index_range[0];
        BOOST_ASSERT(at >= m_index_range[0] && at < m_index_range[1]);
        return { {m_index_range[0], at}, at, {at+1, m_index_range[1]} };
    }

private:
    IndexRange m_index_range;
    CoordRange m_coord_range;
    real_type m_factor;
};

} // namespace s3dmm
