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

#include <list>

namespace s3dmm
{

class IndexBoxSorter
{
public:
    using BBox = BoundingBox<3, unsigned int>;
    using VBBox = std::vector<BBox>;

    void setBoxes(const VBBox& boxes)
    {
        m_boxes = boxes;
        m_valid = false;
    }
    void setDirection(const Vec3d& dir)
    {
        for (size_t i = 0; i < 3; i++)
        {
            m_valid &= (dir[i] * m_dir[i]) > 0;
        }
        m_dir = dir;
    }
    VBBox getBoxes() const
    {
        return m_boxes;
    }
    Vec3d getDirection() const
    {
        return m_dir;
    }
    VBBox getBoxesSorted() const
    {
        sortBoxes();
        return m_boxesSorted;
    }
    VBBox getBoxesSorted(const VBBox& boxes)
    {
        setBoxes(boxes);
        return getBoxesSorted();
    }
    VBBox getBoxesSorted(const VBBox& boxes, const Vec3d& dir)
    {
        setDirection(dir);
        return getBoxesSorted(boxes);
    }

private:
    using BBoxi = BoundingBox<3, int>;
    using VBBoxi = std::vector<BBoxi>;

    Vec3d m_dir{1, 0, 0};
    VBBox m_boxes;
    mutable VBBox m_boxesSorted;
    mutable bool m_valid{false};

    enum BOX_CMP
    {
        BOX_EQUAL,
        BOX_LESS,
        BOX_BIGGER
    };

    void sortBoxes() const
    {
        if (m_valid)
        {
            return;
        }
        m_valid = true;

        VBBoxi boxes;
        std::for_each(m_boxes.begin(), m_boxes.end(), [&](auto& boxIn) {
            BBoxi box;
            if (!boxIn.empty())
            {
                auto vminIn = boxIn.min();
                auto vmaxIn = boxIn.max();
                Vec3i vmin{static_cast<int>(vminIn[0]),
                           static_cast<int>(vminIn[1]),
                           static_cast<int>(vminIn[2])};
                Vec3i vmax{static_cast<int>(vmaxIn[0]),
                           static_cast<int>(vmaxIn[1]),
                           static_cast<int>(vmaxIn[2])};
                for (size_t i = 0; i < 3; i++)
                {
                    if (m_dir[i] < 0)
                    {
                        vmin[i] = -vmin[i];
                        vmax[i] = -vmax[i];
                    }
                }
                box << vmin << vmax;
            }
            boxes.push_back(box);
        });

        auto lessThan = [](const BBoxi& a, const BBoxi& b) {
            BOOST_ASSERT(!a.empty() && !b.empty());
            auto va = a.min();
            auto vb = b.max();
            auto result = va[0] < vb[0] && va[1] < vb[1] && va[2] < vb[2];
            return result;
        };

        std::vector<std::vector<BOX_CMP>> boxCmp(
            boxes.size(), std::vector<BOX_CMP>(boxes.size(), BOX_EQUAL));

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (boxes[i].empty())
            {
                continue;
            }
            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (boxes[j].empty())
                {
                    continue;
                }
                if (lessThan(boxes[i], boxes[j]))
                {
                    boxCmp[i][j] = BOX_LESS;
                    boxCmp[j][i] = BOX_BIGGER;
                }
                else if (lessThan(boxes[j], boxes[i]))
                {
                    boxCmp[j][i] = BOX_LESS;
                    boxCmp[i][j] = BOX_BIGGER;
                }
            }
        }

        std::list<size_t> lidx;
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            auto it = std::find_if(lidx.begin(), lidx.end(), [&](auto x) {
                return boxCmp[i][x] == BOX_LESS;
            });
            lidx.insert(it, i);
        }

        m_boxesSorted.resize(boxes.size());
        size_t idx = 0;
        for (auto i: lidx)
        {
            m_boxesSorted[idx] = m_boxes[i];
            ++idx;
        }
    }
};

} // namespace s3dmm
