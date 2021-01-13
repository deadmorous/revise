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

#include "MeshBoundaryExtractor.hpp"
#include "ScalarOrMultiIndex.hpp"
#include "faceNormalVector.hpp"

namespace s3dmm {

class MeshBoundaryWalker : boost::noncopyable
{
public:
    explicit MeshBoundaryWalker(
            unsigned int boundaryRefine) :
        m_boundaryRefine(boundaryRefine)
    {}

    template<MeshElementType elementType, class F>
    void walkRefinedNodes(
            const MeshBoundaryExtractor::MeshZoneBoundaryData<elementType>& zd,
            F f) const
    {
        using ET = MeshElementTraits<elementType>;
        constexpr const unsigned int N = ET::SpaceDimension;
        using vector_type = ScalarOrMultiIndex_t<N, real_type>;
        for (auto& fd : zd.faces) {
            // Compute outer normal vector to the face
            auto n = faceNormalVector(fd.face, zd.nodes.data());

            // Compute refined element size
            auto size = fd.size / m_boundaryRefine;

            // Get face nodes
            typename ET::Node faceNodes[ET::ElementFaceSize];
            boost::range::transform(fd.face, faceNodes, [&](unsigned int inode) {
                return zd.nodes[inode];
            });

            // Interpolate face nodes
            RefinedMeshElement<N-1, MeshElementTopology::Cube> refined(fd.size, fd.maxIndex*m_boundaryRefine);
            auto refinedNodeCount = refined.count();
            auto interpolatedRefinedSize = refinedNodeCount * N;
            if (m_interpolatedRefined.size() < interpolatedRefinedSize)
                m_interpolatedRefined.resize(interpolatedRefinedSize);
            refined.interpolate(m_interpolatedRefined.data(), N, faceNodes[0].data(), N);
            auto interpolatedRefinedPtr = m_interpolatedRefined.data();

            // Process refined nodes
            for (auto iRefinedNode=0u; iRefinedNode<refinedNodeCount; ++iRefinedNode, interpolatedRefinedPtr+=N) {
                auto& pos = *reinterpret_cast<const vector_type*>(interpolatedRefinedPtr);
                f(pos, n, size);
            }
        }
    }

private:
    unsigned int m_boundaryRefine;
    mutable std::vector<real_type> m_interpolatedRefined;
};

} // s3dmm
