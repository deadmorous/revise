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
#include "FullBlockTreeGenerator.hpp"
#include "ImageToBlockIndexTransform.hpp"

#include <boost/range/algorithm/max_element.hpp>

namespace s3dmm {

template <unsigned int N, class ImageVoxels>
class BlockTreeFromImage
{
public:
    BlockTreeFromImage(
        const MultiIndex<N, unsigned int>& imageSize,
        const ImageVoxels& imageVoxels,
        unsigned int minTreeDepth,
        unsigned int maxTreeDepth,
        const BoundingCube<N, real_type>& pos = BoundingCube<N, real_type>())
        :
        m_imageSize(imageSize),
        m_imageVoxels(imageVoxels),
        m_minTreeDepth(minTreeDepth),
        m_maxTreeDepth(maxTreeDepth),
        m_pos(pos)
    {}

    BlockTree<N> makeBlockTree() const
    {
        auto maxImageSize = *boost::range::max_element(m_imageSize);

        auto sufficientDepth = ImageToBlockIndexTransform<N>::sufficientTreeDepth(m_imageSize);
        auto depth = std::min(sufficientDepth, m_maxTreeDepth);
        auto sufficientVoxelsPerEdge = 1u << sufficientDepth;

        BoundingCube<N, real_type> btPos;
        if (m_pos.size() > 0) {
            auto voxelPhysicalSize = m_pos.size() / maxImageSize;
            auto size = voxelPhysicalSize * sufficientVoxelsPerEdge;
            auto origin = m_pos.center() -
                          ScalarOrMultiIndex<N, real_type>::fromMultiIndex(
                              MultiIndex<N, real_type>::filled(size/2));
            btPos = BoundingCube<N, real_type>(origin, size);
        }
        else {
            auto defaultSize = make_real(2);
            auto origin = ScalarOrMultiIndex<N, real_type>::fromMultiIndex(
                MultiIndex<N, real_type>::filled(-defaultSize/2));
            btPos = BoundingCube<N, real_type>(origin, defaultSize);
        }
        auto result = FullBlockTreeGenerator<N>(m_minTreeDepth, btPos).makeBlockTree();

        ImageToBlockIndexTransform<N> i2b(m_imageSize, depth);

        for (auto& imageIndex : m_imageVoxels)
            result.ensureBlockAt(i2b(imageIndex), depth);
        return result;
    }

    unsigned int maxTreeDepth() const {
        return m_maxTreeDepth;
    }

private:
    MultiIndex<N, unsigned int> m_imageSize;
    ImageVoxels m_imageVoxels;
    unsigned int m_minTreeDepth;
    unsigned int m_maxTreeDepth;
    BoundingCube<N, real_type> m_pos;
};

} // s3dmm
