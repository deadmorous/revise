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

#include "MeshElementTraits.hpp"
#include "MeshElementRefiner.hpp"

#include "SingleElementAllocator.hpp"

#include <vector>
#include <stdexcept>

//#include <unordered_set>
//#include <unordered_map>
#include <set>
#include <map>

#include <boost/noncopyable.hpp>
#include <boost/any.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/iterator/iterator_facade.hpp>

namespace s3dmm {

namespace detail {

template <MeshElementType elementType> struct SimpleFaceExtractor
{
    using ET = MeshElementTraits<elementType>;
    using Face = typename ET::Face;

    static Face extractFace(
            const unsigned int *elementNodeNumbers,
            unsigned int iface,
            std::enable_if_t<ET::SpaceDimension<3, int> = 0)
    {
        auto result = ET::localFaceIndices(iface);
        boost::range::transform(
                    result,
                    result.begin(),
                    [elementNodeNumbers](unsigned int ilocal)
        {
            return elementNodeNumbers[ilocal];
        });
        return result;
    }

    static Face extractFace(
        const unsigned int *elementNodeNumbers,
        unsigned int iface,
        ElementSubtype,
        std::enable_if_t<ET::SpaceDimension<3, int> = 0)
    {
        return extractFace(elementNodeNumbers, iface);
    }

    static Face revertFace(const Face& face) {
        Face result = face;
        std::reverse(result.begin(), result.end());
        return result;
    }
};

template <MeshElementType elementType> struct SpatialFaceExtractor
{
    using ET = MeshElementTraits<elementType>;
    using Face = typename ET::Face;
    static constexpr unsigned int FaceSize = ET::ElementFaceSize;
    static Face extractFace(
            const unsigned int *elementNodeNumbers,
            unsigned int iface,
            std::enable_if_t<ET::SpaceDimension==3, int> = 0)
    {
        auto faceNodeNumbers = ET::localFaceIndices(iface);
        boost::range::transform(
                    faceNodeNumbers,
                    faceNodeNumbers.begin(),
                    [elementNodeNumbers](unsigned int ilocal)
        {
            return elementNodeNumbers[ilocal];
        });
        auto i0 = static_cast<unsigned int>(std::min_element(faceNodeNumbers.begin(), faceNodeNumbers.end()) - faceNodeNumbers.begin());
        Face result;
        for (auto i=0u; i<FaceSize; ++i)
            result[i] = faceNodeNumbers[(i+i0)%FaceSize];
        return result;
    }

    static Face extractFace(
        const unsigned int *elementNodeNumbers,
        unsigned int iface,
        ElementSubtype subtype,
        std::enable_if_t<ET::SpaceDimension==3, int> = 0)
    {
        auto faceNodeNumbers = ET::localFaceIndices(iface, subtype);
        boost::range::transform(
            faceNodeNumbers,
            faceNodeNumbers.begin(),
            [elementNodeNumbers](unsigned int ilocal)
            {
                return ilocal == ~0u? ilocal: elementNodeNumbers[ilocal];
            });
        auto i0 = static_cast<unsigned int>(std::min_element(faceNodeNumbers.begin(), faceNodeNumbers.end()) - faceNodeNumbers.begin());
        Face result;
        for (auto i=0u; i<FaceSize; ++i)
            result[i] = faceNodeNumbers[(i+i0)%FaceSize];
        return result;
    }

    static Face revertFace(const Face& face) {
        Face result = face;
        std::reverse(result.begin()+1, result.end());
        return result;
    }
};

template<unsigned int dim>
struct FaceExtractor {
    template<MeshElementType elementType> using type = SimpleFaceExtractor<elementType>;
};

template<>
struct FaceExtractor<3> {
    template<MeshElementType elementType> using type = SpatialFaceExtractor<elementType>;
};

template <MeshElementType elementType>
using FaceExtractor_t = typename FaceExtractor<MeshElementTraits<elementType>::SpaceDimension>::template type<elementType>;

} // detail

class MeshBoundaryExtractor : boost::noncopyable
{
public:
    template<MeshElementType elementType>
    struct MeshZoneBoundaryFaceData
    {
        static constexpr const int N = MeshElementTraits<elementType>::SpaceDimension;
        MeshElementFace<elementType> face;
        MultiIndex<N-1, unsigned int> maxIndex; // from refiner
        real_type size;                         // from refiner
    };

    template<MeshElementType elementType>
    struct MeshZoneBoundaryData
    {
        std::vector<MeshZoneBoundaryFaceData<elementType>> faces;
        std::vector<MeshElementNode<elementType>> nodes;
    };

    class MeshZoneBoundaryUnifidData
    {
    private:
        MeshElementType m_type;
        boost::any m_data;

    public:
        template<MeshElementType eType>
        explicit MeshZoneBoundaryUnifidData(MeshZoneBoundaryData<eType>&& zoneData) :
            m_type(eType),
            m_data(std::move(zoneData))
        {}
        MeshElementType elementType() const {
            return m_type;
        }
        template<MeshElementType eType>
        const MeshZoneBoundaryData<eType>& data() const {
            return boost::any_cast<const MeshZoneBoundaryData<eType>&>(m_data);
        }
    };

private:
    using MeshBoundaryData = std::vector<MeshZoneBoundaryUnifidData>;

//    template<class K, class V>
//    using fast_map = std::map<K, V>;

//    template<class K>
//    using fast_set = std::set<K>;

    template<class K, class V>
    using fast_map = std::map<K, V, std::less<K>, SingleElementAllocator<std::pair<const K, V>>>;

    template<class K>
    using fast_set = std::set<K, std::less<K>, SingleElementAllocator<K>>;

//    template<class K, class V>
//    using fast_map = std::unordered_map<K, V>;

//    template<class K>
//    using fast_set = std::unordered_set<K>;

    MeshBoundaryData m_data;

    template<MeshElementType elementType, class MeshRefiner>
    class ZoneBoundaryExtractor
    {
    private:
        using ET = MeshElementTraits<elementType>;
        using Face = typename ET::Face;
        static constexpr unsigned int FaceSize = ET::ElementFaceSize;
        using Node = MeshElementNode<elementType>;
        using FaceExtractor = detail::FaceExtractor_t<elementType>;

        template<class MeshProvider>
        static fast_map<Face, unsigned int> extractBoundaryFaces(const typename MeshProvider::MeshElements& zoneElements)
        {
            fast_map<Face, unsigned int> result;
            auto ielement = 0u;
            for (auto& edata : zoneElements) {
                auto subtype = ET::elementSubtype(edata.begin());
                auto faceCount = ET::elementFaceCount(subtype);
                for (auto iface=0u; iface<faceCount; ++iface) {
                    auto face = FaceExtractor::extractFace(edata.begin(), iface, subtype);
                    auto it = result.find(face);
                    if (it != result.end())
                        result.erase(it);
                    else
                        result[FaceExtractor::revertFace(face)] = ielement;
                }
                ++ielement;
            }
            return result;
        }

        static MultiIndex<0, unsigned int> faceMaxIndex(const MultiIndex<1, unsigned int>& /*maxIndex*/, unsigned int /*iface*/) {
            return MultiIndex<0, unsigned int>();
        }

        static MultiIndex<1, unsigned int> faceMaxIndex(const MultiIndex<2, unsigned int>& maxIndex, unsigned int iface) {
            return MultiIndex<1, unsigned int>({maxIndex[1-(iface>>1)]});
        }

        static MultiIndex<2, unsigned int> faceMaxIndex(const MultiIndex<3, unsigned int>& maxIndex, unsigned int iface)
        {
            static const unsigned int i[3][2] = {{2,1}, {0,2}, {0,1}};
            auto i1 = iface >> 1;
            auto i2 = iface & 1;
            return { maxIndex[i[i1][i2]], maxIndex[i[i1][1-i2]] };
        }

    public:
        static constexpr const int N = MeshElementTraits<elementType>::SpaceDimension;

        template<class MeshProvider>
        static MeshZoneBoundaryData<elementType> run(const typename MeshProvider::Zone& zone, const MeshRefiner& meshRefiner)
        {
            MeshZoneBoundaryData<elementType> result;

            // Extract boundary faces
            auto zoneElements = zone.elements();
            auto boundaryFaces = extractBoundaryFaces<MeshProvider>(zoneElements);

            // Generate the numbers for mesh nodes at the boundary;
            // track elements containing boundary faces.
            fast_map<unsigned int, unsigned int> bnodes;    // key=old node number, value=new node number
            fast_set<unsigned int> belements;               // key=number of element containing boundary faces
            for (auto& bfItem : boundaryFaces) {
                for (auto inode : bfItem.first) {
                    if (inode != ~0u) {
                        auto itbnodes = bnodes.find(inode);
                        if (itbnodes == bnodes.end())
                            bnodes[inode] = bnodes.size();
                    }
                }
                belements.insert(bfItem.second);
            }

            // Generate faces
            std::vector<real_type> nodes(zone.nodesPerElement() * N);
            for (auto ielement : belements) {
                auto itZoneElement = zoneElements.begin() + ielement;
                auto nd = nodes.data();
                for (auto nodePos : itZoneElement.elementNodes()) {
                    BOOST_ASSERT(nodePos.size() == N);
                    boost::range::copy(nodePos, nd);
                    nd += N;
                }
                // BUGgy iterator: that won't work: auto elementNodeNum = zoneElements[ielement].begin();
                auto elementNodeNum = itZoneElement->begin();
                auto refined = meshRefiner.refine(nodes.data(), N);
                auto subtype = ET::elementSubtype(elementNodeNum);
                auto faceCount = ET::elementFaceCount(subtype);
                for (auto iface=0u; iface<faceCount; ++iface) {
                    auto face = FaceExtractor::extractFace(elementNodeNum, iface, subtype);
                    if (boundaryFaces.find(FaceExtractor::revertFace(face)) != boundaryFaces.end())
                        result.faces.push_back({face, faceMaxIndex(refined.maxIndex(), iface), refined.size()});
                }
            }

            // Generate nodes
            auto zoneNodes = zone.nodes();
            result.nodes.resize(bnodes.size());
            for (auto nodeItem : bnodes) {
                auto meshNodeData = zoneNodes[nodeItem.first];
                auto& node = result.nodes[nodeItem.second];
                BOOST_ASSERT(meshNodeData.size() == N);
                boost::range::copy(meshNodeData, node.begin());
            }

            // Renumber faces
            for (auto& bface : result.faces) {
                // Merge non-existent node, if any, to neighboring existing node
                // (treat triangular faces as degenerate quads)
                for (auto i=0; i<FaceSize; ++i) {
                    if (bface.face[i] == ~0u) {
                        bface.face[i] = bface.face[(i+1)%FaceSize];
                        BOOST_ASSERT(bface.face[i] != ~0u);
                        break;
                    }
                }
                boost::range::transform(bface.face, bface.face.begin(), [&](unsigned int inode) {
                    return bnodes.at(inode);
                });
            }

            return result;
        }
    };

    template<class MeshProvider, MeshElementType elementType>
    void processZone(const typename MeshProvider::Zone& zone, const MeshElementRefinerParam& meshRefinerParam)
    {
        if (meshRefinerParam.needRefiner()) {
            auto meshRefiner = meshRefinerParam.template refiner<elementType>();
            m_data.emplace_back(ZoneBoundaryExtractor<elementType, decltype(meshRefiner)>::template run<MeshProvider>(zone, meshRefiner));
        }
        else {
            auto meshRefiner = meshRefinerParam.template trivialRefiner<elementType>();
            m_data.emplace_back(ZoneBoundaryExtractor<elementType, decltype(meshRefiner)>::template run<MeshProvider>(zone, meshRefiner));
        }
    }

public:

    template<class MeshProvider>
    MeshBoundaryExtractor(MeshProvider& meshProvider, const MeshElementRefinerParam& meshRefinerParam)
    {
        std::vector<unsigned int> vars = meshProvider.coordinateVariables();
        auto cache = meshProvider.makeCache();
        for (auto zone : meshProvider.zones(cache, vars)) {
            switch (zone.elementType()) {
            case MeshElementType::Triangle:
                if (vars.size() != 2)
                    throw std::runtime_error("Invalid dimension in the mesh file");
                processZone<MeshProvider, MeshElementType::Triangle>(zone, meshRefinerParam);
                break;
            case MeshElementType::Quad:
                if (vars.size() != 2)
                    throw std::runtime_error("Invalid dimension in the mesh file");
                processZone<MeshProvider, MeshElementType::Quad>(zone, meshRefinerParam);
                break;
            case MeshElementType::Tetrahedron:
                if (vars.size() != 3)
                    throw std::runtime_error("Invalid dimension in the mesh file");
                processZone<MeshProvider, MeshElementType::Tetrahedron>(zone, meshRefinerParam);
                break;
            case MeshElementType::Hexahedron:
                if (vars.size() != 3)
                    throw std::runtime_error("Invalid dimension in the mesh file");
                processZone<MeshProvider, MeshElementType::Hexahedron>(zone, meshRefinerParam);
                break;
            }
        }
    }

    template<class It>
    using range = boost::iterator_range<It>;

    using ZonesIterator = MeshBoundaryData::const_iterator;
    using Zones = range<ZonesIterator>;

    Zones zones() const {
        return { m_data.begin(), m_data.end() };
    }
};

} // s3dmm
