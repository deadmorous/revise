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

#include "IndexBoxSorter.hpp"
#include "IndexCubeSplitter.hpp"

namespace s3vs
{

class IndexCubeSplitStorage
{
public:
    using BBox = s3dmm::IndexBoxSorter::BBox;
    using VBBox = s3dmm::IndexBoxSorter::VBBox;

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
    void setDirection(const s3dmm::Vec3d& dir)
    {
        m_dir = dir;
    }
    s3dmm::Vec3d getDirection() const
    {
        return m_dir;
    }
    VBBox getBoxesSorted(unsigned int level)
    {
        resizeSortedBoxes(level);
        return m_sortedBoxes[level].getBoxesSorted(m_dir);
    }
    VBBox getBoxes(unsigned int level)
    {
        resizeSortedBoxes(level);
        return m_sortedBoxes[level].getBoxes();
    }

private:
    class SortedBoxes
    {
    public:
        SortedBoxes(unsigned int level, unsigned int splitCount)
        {
            s3dmm::IndexCubeSplitter splitter;
            auto boxes = splitter.split(level, splitCount);
            decltype(boxes) b;
            for (auto box: boxes)
                if (!box.empty() && box.size()[0] && box.size()[1] && box.size()[2])
                    b.push_back(box);
            m_sorter.setBoxes(b);
        }
        VBBox getBoxesSorted(const s3dmm::Vec3d& dir)
        {
            m_sorter.setDirection(dir);
            return m_sorter.getBoxesSorted();
        }
        VBBox getBoxes() const
        {
            return m_sorter.getBoxes();
        }

    private:
        s3dmm::IndexBoxSorter m_sorter;
    };
    std::vector<SortedBoxes> m_sortedBoxes;
    unsigned int m_splitCount{1};
    s3dmm::Vec3d m_dir{1, 0, 0};

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
