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
#include "s3dmm/IncMultiIndex.hpp"

namespace s3dmm
{

class SplitNumber
{
public:
    SplitNumber(unsigned int number, unsigned int split) :
      m_number(number), m_split(split)
    {
        m_count = m_number / m_split;
        m_remainder = m_number % m_split;
    }

    unsigned int operator[](unsigned int idx) const
    {
        return idx < m_remainder ? m_count + 1 : m_count;
    }

private:
    unsigned int m_number;
    unsigned int m_split;
    unsigned int m_count;
    unsigned int m_remainder;
};

class IndexCubeSplitter
{
public:
    using BBox = BoundingBox<3, unsigned int>;
    std::vector<BBox> split(unsigned int level, unsigned int splitCount_)
    {
        std::vector<BBox> res(splitCount_);

        if (!splitCount_)
        {
            return res;
        }

        const unsigned int splitCountMax = 1 << (3 * level);
        const unsigned int splitCount = std::min(splitCount_, splitCountMax);
        const unsigned int splitCountEdge = 1 << level;

        if (splitCount <= splitCountEdge)
        {
            // simply split into slices in the 0-st dimension
            SplitNumber sn(splitCountEdge, splitCount);
            Vec3u vmin{0, 0, 0};
            for (auto i = 0U; i < splitCount; ++i)
            {
                res[i] << vmin;
                vmin[0] += sn[i];
                res[i] << Vec3u({vmin[0], splitCountEdge, splitCountEdge});
            }
            return res;
        }

        Vec3u range{1, 1, 1};
        Vec3u step{splitCountEdge, splitCountEdge, splitCountEdge};
        {
            auto i = 0U;
            auto remainder = splitCount;
            for (; remainder >= splitCountEdge; remainder /= splitCountEdge)
            {
                range[i] = splitCountEdge;
                step[i] = 1;
                ++i;
            }
            if (i < 3U)
            {
                while (remainder > 1)
                {
                    range[i] <<= 1;
                    remainder >>= 1;
                }
                step[i] = splitCountEdge / range[i];
            }
        }

        Vec3u index{0, 0, 0};
        for (auto i = 0;; ++i)
        {
            auto v = elementwiseMultiply(index, step);
            res[i] << v << (v + step);
            if (!incMultiIndex(index, Vec3u{0, 0, 0}, range))
            {
                break;
            }
        }
        return res;
    }
};

} // namespace s3dmm
