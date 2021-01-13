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

#include "BlockTree.hpp"
#include "ProgressReport.hpp"
#include "IncMultiIndex.hpp"

#include <fstream>


namespace s3dmm {

template <unsigned int N>
class FullBlockTreeGenerator
{
public:
    FullBlockTreeGenerator(
            unsigned int maxTreeDepth,
            BoundingCube<N, real_type> pos)
        :
        m_maxTreeDepth(maxTreeDepth),
        m_pos(pos)
    {}

    FullBlockTreeGenerator(
            unsigned int maxTreeDepth,
            BoundingBox<N, real_type> pos)
        :
        m_maxTreeDepth(maxTreeDepth),
        m_pos(pos)
    {}

    BlockTree<N> makeBlockTree() const
    {
        REPORT_PROGRESS_STAGES();
        REPORT_PROGRESS_STAGE("Generate full block tree");
        BlockTree<N> result(m_pos);
        auto cellCount = 1u << m_maxTreeDepth;
        BOOST_ASSERT(cellCount != 0);
        MultiIndex<N, unsigned int> i;
        do {
            result.ensureBlockAt(i, m_maxTreeDepth);
        }
        while (incMultiIndex(i, cellCount));
        return result;
    }

    unsigned int maxTreeDepth() const {
        return m_maxTreeDepth;
    }

private:
    unsigned int m_maxTreeDepth;
    BoundingCube<N, real_type> m_pos;
};

} // s3dmm
