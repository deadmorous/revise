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

class BlockSorter
{
public:
    using BBox = BoundingBox<3, unsigned int>;
    using v_Vec3u = std::vector<Vec3u>;

    void setBoundingBox(const BBox& box)
    {
        m_valid &= box.empty() == m_box.empty()
                   && box.max() == m_box.max()
                   && box.min() == m_box.min();
        m_box = box;
    }
    BBox getBoundingBox() const
    {
        return m_box;
    }
    void setEye(const Vec3d& eye)
    {
        m_valid &= eye == m_eye;
        m_eye = eye;
    }
    Vec3d getEye() const
    {
        return m_eye;
    }
    void setCenter(const Vec3d& center)
    {
        m_valid &= center == m_center;
        m_center = center;
    }
    Vec3d getCenter() const
    {
        return m_center;
    }

    const v_Vec3u& getSortedBlocks() const
    {
        makeValid();
        return m_result;
    }

    const v_Vec3u& getSortedBlocks(
        const BBox& box, const Vec3d& eye, const Vec3d& center)
    {
        setBoundingBox(box);
        setEye(eye);
        setCenter(center);
        return getSortedBlocks();
    }

    static std::vector<Vec3u> sortBlocks(
        const std::vector<BBox>& inBox, const Vec3d& dir)
    {
        std::vector<std::pair<Vec3u, double>> blocks;

        const auto dx = make_real(0.5);
        for (const auto& box: inBox)
        {
            if (box.empty())
            {
                continue;
            }

            for (auto index = box.min();;)
            {
                Vec3d r({index[0] + dx, index[1] + dx, index[2] + dx});
                auto z = dir * r;
                blocks.push_back(std::make_pair(index, z));
                if (!incMultiIndex(index, box.min(), box.max()))
                {
                    break;
                }
            }
        }

        std::sort(
            blocks.begin(), blocks.end(), [](const auto& x1, const auto& x2) {
                return x1.second < x2.second;
            });

        std::vector<Vec3u> res(blocks.size());
        for (size_t i = 0; i < res.size(); ++i)
        {
            res[i] = blocks[i].first;
        }
        return res;
    }

private:
    BBox m_box;
    mutable std::vector<Vec3u> m_result;
    mutable bool m_valid{false};
    Vec3d m_eye{1, 0, 0};
    Vec3d m_center{0, 0, 0};

    void makeValid() const
    {
        if (m_valid)
            return;
        m_valid = true;
        // TODO: make it better
        m_result = sortBlocks({{m_box}}, m_eye - m_center);
    }
};

} // namespace s3dmm
