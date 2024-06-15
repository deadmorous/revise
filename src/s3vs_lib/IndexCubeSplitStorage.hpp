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

#include "BackToFrontOrder.hpp"
#include "IndexCubeSplitter2.hpp"

namespace s3vs
{

class IndexCubeSplitStorage
{
public:
    using BBox = s3dmm::BoundingBox<3, unsigned int>;
    using VBBox = std::vector<BBox>;

    void setSplitCount(unsigned int splitCount)
    {
        if (m_splitCount != splitCount)
            m_sortedBoxes.clear();
        m_splitCount = splitCount;
    }
    unsigned int getSplitCount() const
    {
        return m_splitCount;
    }
    void setEyePosition(const s3dmm::Vec3d& eye)
    {
        m_eye = eye;
    }
    s3dmm::Vec3d getEyePosition() const
    {
        return m_eye;
    }
    VBBox getBoxesSorted(unsigned int level)
    {
        resizeSortedBoxes(level);
        return m_sortedBoxes[level].getBoxesSorted(m_eye);
    }

private:
    class SortedBoxes
    {
    public:
        SortedBoxes(unsigned int level, unsigned int splitCount)
            : m_splitter{ level, splitCount, default_bbox() }
        {}

        VBBox getBoxesSorted(const s3dmm::Vec3d& eye)
        {
            auto result = VBBox{};
            result.reserve(m_splitter.split_count());
            auto b2fo = s3dmm::BackToFrontOrder{m_splitter};
            for (const auto& block: b2fo.range(eye))
                result.push_back(block);
            return result;
        }

    private:
        using CoordBBox = s3dmm::BoundingBox<3, s3dmm::real_type>;
        static CoordBBox default_bbox() noexcept
        {
            return CoordBBox{}
                << s3dmm::Vec3d{ -5, -5, -5 }
                << s3dmm::Vec3d{  5,  5,  5 };
        }

        s3dmm::IndexCubeSplitter2<3> m_splitter;
    };
    std::vector<SortedBoxes> m_sortedBoxes;
    unsigned int m_splitCount{1};
    s3dmm::Vec3d m_eye{0, 0, 10};

    void resizeSortedBoxes(unsigned int level)
    {
        while (level >= m_sortedBoxes.size())
        {
            m_sortedBoxes.push_back(
                SortedBoxes(m_sortedBoxes.size(), m_splitCount));
        }
    }
};

} // namespace s3vs
