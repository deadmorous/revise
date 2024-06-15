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

#include "BoundingBox.hpp"
#include "ScalarOrMultiIndex.hpp"
#include "UniformIndexRangeSplitter.hpp"

namespace s3dmm
{

template <unsigned int N>
class MacroBlock
{
public:

    using IndexVecTraits = ScalarOrMultiIndex<N, unsigned int>;
    using Block = IndexVecTraits::type;
    using IndexVec = IndexVecTraits::type;
    using Splitter = UniformIndexRangeSplitter;
    using IndexBox = BoundingBox<N, unsigned int>;
    using CoordBox = BoundingBox<N, real_type>;

    static constexpr auto dim = N;

    explicit MacroBlock(const IndexBox& index_box,
                        const CoordBox& coord_box):
        m_index_box{ index_box },
        m_coord_box{ coord_box }
    {}

    Block at(const IndexVec& pos) const noexcept
    { return pos; }

    IndexVec begin_index() const noexcept
    { return m_index_box.min(); }

    IndexVec end_index() const noexcept
    { return m_index_box.max(); }

    Splitter index_range_splitter(unsigned int axis) const noexcept
    {
        BOOST_ASSERT(axis < dim);
        return { m_index_box.range(axis), m_coord_box.range(axis) };
    }

private:
    IndexBox m_index_box;
    CoordBox m_coord_box;
};



} // namespace s3dmm
