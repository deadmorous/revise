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

#include "Metadata.hpp"
#include "MeshElementTraits.hpp"
#include "elementBoundingBox.hpp"
#include "ElementApprox.hpp"
#include "binary_io.hpp"

#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/transform.hpp>

#include <experimental/filesystem>
#include <fstream>

// #define DEBUG_ELEMENT_MAPPER_STATS

namespace s3dmm {

namespace BlockTreeIMappedFieldsFromFile_detail {

template<unsigned int N, class MeshProvider>
struct MapElementParam
{
    using BT = typename Metadata<N>::BT;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    real_type *weightValues;
    const BoundingCube<N, real_type>& subtreePos;
    const BlockTreeNodes<N, BT>& subtreeNodes;
    const typename MeshProvider::Zone& zone;
    unsigned int ielement;
    const typename MeshProvider::ElementNodes& elementNodes;
    const NodeIndex& imin;
    const NodeIndex& imax;
    real_type gridSize;
    BinaryWriter& writer;
    std::size_t& badInside;
    std::size_t& badOutside;
};

template<unsigned int N, class MeshProvider>
struct MapZoneParam
{
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;

    const Metadata<N>& metadata;
    real_type *weightValues;
    const BlockId& subtreeRoot;
    const BoundingCube<N, real_type>& subtreePos;
    const BlockTreeNodes<N, BT>& subtreeNodes;
    const typename MeshProvider::Zone& zone;
    const std::vector<BoundingCube<N, real_type>>& processedAreas;
    std::ostream& fieldMap;
};

template<MeshElementType elementType, class ElementData>
inline Vec<MeshElementTraits<elementType>::SpaceDimension+1, unsigned int>
getSimplexGlobalNodeNumbers(
    const ElementData& element, unsigned int isub, unsigned int isimp)
{
    using ET = MeshElementTraits<elementType>;
    constexpr auto N = MeshElementTraits<elementType>::SpaceDimension;
    Vec<N+1, unsigned int> result;
    boost::range::transform(
        ET::subtypeLocalSimplexIndices(isub, isimp), result.begin(),
        [&element](unsigned int ilocal) {
            return element[ilocal];
        });
    return result;
}

using const_real_range = boost::iterator_range<const real_type*>;

template<unsigned int N, class MeshNodes>
inline Vec<N+1, const_real_range> getSimplexElementNodes(
    const MeshNodes& meshNodes, const Vec<N+1, unsigned int>& simplexGlobalNodeNumbers)
{
    Vec<N+1, const_real_range> result;
    boost::range::transform(
        simplexGlobalNodeNumbers, result.begin(), [&meshNodes](unsigned int iglobal) {
            return meshNodes[iglobal];
        });
    return result;
}

template<MeshElementType elementType>
struct MeshSingleElementMapper
{
    static constexpr const unsigned int N = MeshElementTraits<elementType>::SpaceDimension;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    template<class MeshProvider>
    static void mapSingleElement(const MapElementParam<N, MeshProvider>& p)
    {
        auto elementApprox = makeElementApprox<N, elementType>(p.elementNodes);

        p.subtreeNodes.walkIndexBoxNodes(p.imin, p.imax, [&](const NodeIndex& nodeIndex, std::size_t nodeNumber) {
            if (p.weightValues[nodeNumber] > 0)
                return;
            auto nodePos = reinterpret_cast<const Vec<N,real_type>&>(p.subtreePos.min()) + nodeIndex*p.gridSize;
            Vec<N,real_type> param;
            auto pr = elementApprox.param(param, nodePos);
            if (pr.second) {
                if (pr.first)
                    p.writer
                        << p.ielement << static_cast<unsigned char>(0)
                        << static_cast<unsigned int>(nodeNumber) << param;
            }
            else {
                ++(pr.first? p.badInside: p.badOutside);
            }
        });
    }
};

template<MeshElementType elementType, ElementSubtype subtype>
struct MeshElementSubtypeMapper
{
    using ET = MeshElementTraits<elementType>;
    static constexpr const unsigned int N = ET::SpaceDimension;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    template<class MeshProvider>
    static void mapElementSubtype(const MapElementParam<N, MeshProvider>& p)
    {
        // Compute nodes for all simplex elements;
        // Create approximator for each simplex
        constexpr auto isub = ET::subtypeIndex(subtype);
        constexpr auto nsimp = ET::subtypeSimplexCount(isub);
        Vec<N+1, const_real_range> nodes[nsimp];
        constexpr auto simplexElementType = simplex_element_type_v<N>;
        ElementApprox<N, simplexElementType> elementApprox[nsimp];
        // Note: The line below won't work (buggy mesh provider iterators)
        // auto& element = p.zone.elements()[p.ielement];
        auto itElement = p.zone.elements().begin() + p.ielement;
        auto& element = *itElement;
        for (auto isimp=0u; isimp<nsimp; ++isimp) {
            auto simpGlobalNodeNumbers = getSimplexGlobalNodeNumbers<elementType>(
                element, isub, isimp);
            auto& simpNodes = nodes[isimp];
            simpNodes = getSimplexElementNodes<N>(p.zone.nodes(), simpGlobalNodeNumbers);
            elementApprox[isimp] = makeElementApprox<N, simplexElementType>(simpNodes);
        }

        auto encodeSimplex = [](unsigned int isimp) {
            BOOST_ASSERT(isub+1 < 0xf);
            BOOST_ASSERT(isimp < 0xf);
            return static_cast<unsigned char>(((isub+1)<<4) + isimp);
        };

        // Process all simplex elements on the grid
        p.subtreeNodes.walkIndexBoxNodes(p.imin, p.imax, [&](const NodeIndex& nodeIndex, std::size_t nodeNumber) {
            if (p.weightValues[nodeNumber] > 0)
                return;
            auto nodePos = reinterpret_cast<const Vec<N,real_type>&>(p.subtreePos.min()) + nodeIndex*p.gridSize;
            for (auto isimp=0u; isimp<nsimp; ++isimp) {
                Vec<N,real_type> param;
                auto pr = elementApprox[isimp].param(param, nodePos);
                if (pr.second) {
                    if (pr.first)
                        p.writer
                            << p.ielement << encodeSimplex(isimp)
                            << static_cast<unsigned int>(nodeNumber) << param;
                }
                else {
                    ++(pr.first? p.badInside: p.badOutside);
                }
            }
        });
    }
};

template<MeshElementType elementType, ElementSubtype subtype, class = void>
struct MeshElementMapper;

template<MeshElementType elementType, ElementSubtype subtype>
struct MeshElementMapper<
    elementType, subtype,
    std::enable_if_t<!is_element_subtype_v<elementType, subtype>, void>>
{
    static constexpr const unsigned int N = MeshElementTraits<elementType>::SpaceDimension;
    template<class MeshProvider>
    static void mapElement(const MapElementParam<N, MeshProvider>& p) {
        MeshSingleElementMapper<elementType>::mapSingleElement(p);
    }
};

template<MeshElementType elementType, ElementSubtype subtype>
struct MeshElementMapper<
    elementType, subtype,
    std::enable_if_t<is_element_subtype_v<elementType, subtype>, void>>
{
    static constexpr const unsigned int N = MeshElementTraits<elementType>::SpaceDimension;
    template<class MeshProvider>
    static void mapElement(const MapElementParam<N, MeshProvider>& p) {
        MeshElementSubtypeMapper<elementType, subtype>::mapElementSubtype(p);
    }
};

template<MeshElementType elementType, ElementSubtype subtype, unsigned int N, class MeshProvider>
void mapElement(const MapElementParam<N, MeshProvider>& p) {
    MeshElementMapper<elementType, subtype>::mapElement(p);
}

#ifdef DEBUG_ELEMENT_MAPPER_STATS
template<unsigned int N>
class ElementMapperStats
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    ~ElementMapperStats() {
        flush();
    }

    void newBlock(const BlockId& subtreeRoot)
    {
        flush();
        m_subtreeRoot = subtreeRoot;
    }

    void newEmptyElement() {
        ++m_blockStats.emptyElementCount;
    }

    template<class ElementNodes>
    void newFilledElement(
        const NodeIndex& imin, const NodeIndex& imax,
        unsigned int ielement, const BoundingBox<N, real_type>& elementBox, const ElementNodes& elementNodes)
    {
        auto grid = imax - imin;
        auto gridSize = 1u;
        for (auto d=0u; d<N; ++d)
            gridSize *= grid[d];
        BOOST_ASSERT(gridSize > 0);
        if (m_blockStats.filledElementCount == 0) {
            m_blockStats.minGridSize = m_blockStats.maxGridSize = gridSize;
            m_blockStats.minGrid = m_blockStats.maxGrid = grid;
        }
        else if (m_blockStats.minGridSize > gridSize) {
            m_blockStats.minGridSize = gridSize;
            m_blockStats.minGrid = grid;
        }
        else if (m_blockStats.maxGridSize < gridSize) {
            m_blockStats.maxGridSize = gridSize;
            m_blockStats.maxGrid = grid;

            m_blockStats.ielement = ielement;
            m_blockStats.elementBox = elementBox;
            m_blockStats.elementNodes.clear();
            for (auto& node : elementNodes) {
                for (auto d=0u; d<N; ++d)
                    m_blockStats.elementNodes.push_back(node[d]);
            }
        }
        ++m_blockStats.filledElementCount;
        m_blockStats.totalGridPoints += gridSize;
    }

    static ElementMapperStats& instance() {
        static ElementMapperStats result;
        return result;
    }

private:
    ElementMapperStats() : m_os("element_mapper_stats.txt")
    {
        printHeaders();
    }

    void printHeaders()
    {
        auto printNheaders = [this](const char *prefix, char label) {
            for (auto d=0u; d<N; ++d)
                m_os << '\t' << prefix << char(label+d);
        };

        m_os << "level"
                "\tblock_idx";
        printNheaders("block_", 'x');
        m_os << "\tempty_count"
                "\tfilled_count"
                "\tfill_rate"
                "\tmin_grid_size"
                "\tmax_grid_size"
                "\ttotal_grid_points"
                "\tpoints_per_element";
        printNheaders("min_grid_", 'x');
        printNheaders("max_grid_", 'x');
        m_os << "\tielement";
        printNheaders("box_min_", 'x');
        printNheaders("box_max_", 'x');
        m_os << "\tnodes";
        m_os << std::endl;
    }

    void flush()
    {
        if (m_blockStats.emptyElementCount + m_blockStats.filledElementCount > 0) {
            auto printNvalues = [this](const auto *data) {
                for (auto d=0u; d<N; ++d)
                    m_os << '\t' << data[d];
            };

            auto fillRate =
                static_cast<double>(m_blockStats.filledElementCount) /
                    (m_blockStats.filledElementCount + m_blockStats.emptyElementCount);
            auto pointsPerElement =
                static_cast<double>(m_blockStats.totalGridPoints) /
                    m_blockStats.filledElementCount;

            m_os
                << m_subtreeRoot.level << '\t'
                << m_subtreeRoot.index << '\t';
            printNvalues(m_subtreeRoot.location.data());
            m_os
                << '\t'
                << m_blockStats.emptyElementCount << '\t'
                << m_blockStats.filledElementCount << '\t'
                << fillRate << '\t'
                << m_blockStats.minGridSize << '\t'
                << m_blockStats.maxGridSize << '\t'
                << m_blockStats.totalGridPoints << '\t'
                << pointsPerElement;
            printNvalues(m_blockStats.minGrid.data());
            m_os << '\t';
            printNvalues(m_blockStats.maxGrid.data());
            m_os << '\t' << m_blockStats.ielement;
            printNvalues(m_blockStats.elementBox.min().data());
            printNvalues(m_blockStats.elementBox.max().data());
            for (auto & x : m_blockStats.elementNodes)
                m_os << '\t' << x;
            m_os << std::endl;
        }
        m_blockStats = BlockStats();
    }

    BlockId m_subtreeRoot;
    struct BlockStats
    {
        unsigned int emptyElementCount = 0;
        unsigned int filledElementCount = 0;
        unsigned int minGridSize = 0;
        unsigned int maxGridSize = 0;
        unsigned int totalGridPoints = 0;
        NodeIndex minGrid;
        NodeIndex maxGrid;

        // Worst element data
        unsigned int ielement = 0;
        BoundingBox<N, real_type> elementBox;
        std::vector<real_type> elementNodes;
    };
    BlockStats m_blockStats;
    std::ofstream m_os;
};
#endif // DEBUG_ELEMENT_MAPPER_STATS

template <unsigned int N, class MeshProvider>
struct ZoneMapper
{
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    template<MeshElementType elementType>
    static void mapZone(const MapZoneParam<N, MeshProvider>& p)
    {
        using ET = MeshElementTraits<elementType>;

        BinaryWriter writer(p.fieldMap);
        auto maxLevel = p.subtreeRoot.level + p.subtreeNodes.maxDepth();
        auto& blockTree = p.metadata.blockTree();
        BoundingBox<N, real_type> subtreeBox;
        subtreeBox << p.subtreePos.min() << p.subtreePos.max();

        auto elements = p.zone.elements();
        std::size_t badInside = 0;
        std::size_t badOutside = 0;

        auto gridSize = p.subtreePos.size() / (1 << p.subtreeNodes.maxDepth());
        const auto ione = NodeIndex::filled(1u);

        auto ielement = 0u;
        for (auto it=elements.begin(); it!=elements.end(); ++it, ++ielement) {
            auto elementNodes = it.elementNodes();

            auto elementBox = elementBoundingBox<N>(elementNodes) & subtreeBox;

            if (!elementBox.empty() &&
                !std::any_of(p.processedAreas.begin(), p.processedAreas.end(), [&](const auto& bc) {
                    return bc.contains(elementBox);
                }))
            {
                auto minPos = ((elementBox.min()-p.subtreePos.min())/gridSize);
                auto maxPos = ((elementBox.max()-p.subtreePos.min())/gridSize);
                auto imin = ScalarOrMultiIndex<N, real_type>::toMultiIndex(minPos).template convertTo<BlockTreeNodeCoord>();
                auto imax = ScalarOrMultiIndex<N, real_type>::toMultiIndex(maxPos).template convertTo<BlockTreeNodeCoord>() + ione;

                auto subtype = ET::elementSubtype(it->begin());

#ifdef DEBUG_ELEMENT_MAPPER_STATS
                ElementMapperStats<N>::instance().newFilledElement(
                    imin, imax, ielement, elementBox, elementNodes);
                continue;
#endif // DEBUG_ELEMENT_MAPPER_STATS

                MapElementParam<N, MeshProvider> pe = {
                    p.weightValues, p.subtreePos, p.subtreeNodes, p.zone,
                    ielement, elementNodes, imin, imax,
                    gridSize, writer, badInside, badOutside
                };

                switch (subtype) {
                    case ElementSubtype::Triangle:
                        mapElement<elementType, ElementSubtype::Triangle>(pe);
                        break;
                    case ElementSubtype::Quad:
                        mapElement<elementType, ElementSubtype::Quad>(pe);
                        break;
                    case ElementSubtype::Tetrahedron:
                        mapElement<elementType, ElementSubtype::Tetrahedron>(pe);
                        break;
                    case ElementSubtype::Pyramid:
                        mapElement<elementType, ElementSubtype::Pyramid>(pe);
                        break;
                    case ElementSubtype::Prism:
                        mapElement<elementType, ElementSubtype::Prism>(pe);
                        break;
                    case ElementSubtype::Hexahedron:
                        mapElement<elementType, ElementSubtype::Hexahedron>(pe);
                        break;
                }
            }
#ifdef DEBUG_ELEMENT_MAPPER_STATS
            else
                ElementMapperStats<N>::instance().newEmptyElement();
#endif // DEBUG_ELEMENT_MAPPER_STATS
        }
        writer << ~0u;

        if (badInside + badOutside > 0) {
            auto elementCount = elements.size() / ET::ElementNodeCount;
            std::cout << "WARNING: some bad elements in the zone have been discarded: "
                      << badInside << " in + " << badOutside << " out = " << badInside + badOutside
                      << " of " << elementCount
                      << " (" << 100*static_cast<real_type>(badInside) / elementCount
                      << "% + " << 100*static_cast<real_type>(badOutside) / elementCount
                      << "% = " << 100*static_cast<real_type>(badInside + badOutside) / elementCount
                      << "%)"
                      << std::endl;
        }
    }
};

template<unsigned int N>
struct ZoneMapperDispatch;

template<>
struct ZoneMapperDispatch<1>
{
    static constexpr unsigned int N = 1;

    template<class MeshProvider>
    static void dispatch(const MapZoneParam<N, MeshProvider>&)
    {}
};

template<>
struct ZoneMapperDispatch<2>
{
    static constexpr unsigned int N = 2;
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    template<class MeshProvider>
    static void dispatch(const MapZoneParam<N, MeshProvider>& p)
    {
        switch (p.zone.elementType()) {
            case MeshElementType::Triangle:
                ZoneMapper<N, MeshProvider>::template mapZone<MeshElementType::Triangle>(p);
                break;
            case MeshElementType::Quad:
                ZoneMapper<N, MeshProvider>::template mapZone<MeshElementType::Quad>(p);
                break;
            default:
                throw std::runtime_error("ZoneMapperDispatch::dispatch() failed: Unsupported element type");
        }
    }
};

template<>
struct ZoneMapperDispatch<3>
{
    static constexpr unsigned int N = 3;
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    template<class MeshProvider>
    static void dispatch(const MapZoneParam<N, MeshProvider>& p)
    {
#ifdef DEBUG_ELEMENT_MAPPER_STATS
        ElementMapperStats<N>::instance().newBlock(p.subtreeRoot);
#endif // DEBUG_ELEMENT_MAPPER_STATS

        switch (p.zone.elementType()) {
            case MeshElementType::Tetrahedron:
                ZoneMapper<N, MeshProvider>::template mapZone<MeshElementType::Tetrahedron>(p);
                break;
            case MeshElementType::Hexahedron:
                ZoneMapper<N, MeshProvider>::template mapZone<MeshElementType::Hexahedron>(p);
                break;
            default:
                throw std::runtime_error("ZoneMapperDispatch::dispatch() failed: Unsupported element type");
        }
    }
};



template<unsigned int N, class MeshProvider>
struct GenerateZoneParam
{
    const Metadata<N>& metadata;
    std::vector<std::vector<real_type>>& fieldValues;
    std::vector<std::vector<real_type>>& weightValues;
    const typename MeshProvider::Zone& zone;
    std::istream& fieldMap;
};

template <unsigned int N, class MeshProvider>
struct ZoneGenerator
{
    template<MeshElementType elementType>
    static void generateZone(const GenerateZoneParam<N, MeshProvider>& p)
    {
        auto decodeSimplex = [](unsigned char s) {
            BOOST_ASSERT(s != 0);
            unsigned int isimp = s & 0xf;
            unsigned int isub = (s >> 4) - 1;
            return std::make_pair(isub, isimp);
        };

        auto fieldCount = p.fieldValues.size();
        BOOST_ASSERT(p.weightValues.size() == fieldCount);
        std::vector<real_type> buf(fieldCount);
        BinaryReader reader(p.fieldMap);
        auto elements = p.zone.elements();
        while (true) {
            auto ielement = reader.read<unsigned int>();
            if (ielement == ~0u)
                break;
            auto simplexInfo = reader.read<unsigned char>();
            auto nodeNumber = reader.read<unsigned int>();
            auto param = reader.read<Vec<N,real_type>>();
            if (simplexInfo == 0) {
                // Not a simplex, process as one element
                auto elementIter = elements.begin() + ielement;
                auto elementNodes = elementIter.elementNodes();
                ElementApprox<N, elementType>::approx(buf.data(), elementNodes, param);
            }
            else {
                // Simplex, process as subelement
                auto [isub, isimp] = decodeSimplex(simplexInfo);

                // Note: The line below won't work (buggy mesh provider iterators)
                // auto& element = p.zone.elements()[ielement];
                auto itElement = p.zone.elements().begin() + ielement;
                auto& element = *itElement;
                auto simpGlobalNodeNumbers = getSimplexGlobalNodeNumbers<elementType>(element, isub, isimp);
                auto simpNodes = getSimplexElementNodes<N>(p.zone.nodes(), simpGlobalNodeNumbers);
                constexpr auto simplexElementType = simplex_element_type_v<N>;
                ElementApprox<N, simplexElementType>::approx(buf.data(), simpNodes, param);
            }
            for (auto ifield=0u; ifield<fieldCount; ++ifield) {
                p.fieldValues[ifield][nodeNumber] = buf[ifield];
                p.weightValues[ifield][nodeNumber] = make_real(1);
            }
        }
    }
};

template<unsigned int N>
struct ZoneGeneratorDispatch;

template<>
struct ZoneGeneratorDispatch<1>
{
    static constexpr const unsigned int N = 1;
    template<class MeshProvider>
    static void dispatch(const GenerateZoneParam<N, MeshProvider>&)
    {}
};

template<>
struct ZoneGeneratorDispatch<2>
{
    static constexpr const unsigned int N = 2;
    template<class MeshProvider>
    static void dispatch(const GenerateZoneParam<N, MeshProvider>& p)
    {
        switch (p.zone.elementType()) {
            case MeshElementType::Triangle:
                ZoneGenerator<N, MeshProvider>::template generateZone<MeshElementType::Triangle>(p);
                break;
            case MeshElementType::Quad:
                ZoneGenerator<N, MeshProvider>::template generateZone<MeshElementType::Quad>(p);
                break;
            default:
                throw std::runtime_error("ZoneGeneratorDispatch::dispatch() failed: Unsupported element type");
        }
    }
};

template<>
struct ZoneGeneratorDispatch<3>
{
    static constexpr const unsigned int N = 3;
    template<class MeshProvider>
    static void dispatch(const GenerateZoneParam<N, MeshProvider>& p)
    {
        switch (p.zone.elementType()) {
            case MeshElementType::Tetrahedron:
                ZoneGenerator<N, MeshProvider>::template generateZone<MeshElementType::Tetrahedron>(p);
                break;
            case MeshElementType::Hexahedron:
                ZoneGenerator<N, MeshProvider>::template generateZone<MeshElementType::Hexahedron>(p);
                break;
            default:
                throw std::runtime_error("ZoneGeneratorDispatch::dispatch() failed: Unsupported element type");
        }
    }
};

} // BlockTreeIMappedFieldsFromFile_detail



// Note: IMappedFields means "interpolated mapped fields",
// according to the algorithm used to compute field values at block tree nodes.
template<unsigned int N, class MeshProvider, class MeshRefinerParam>
class BlockTreeIMappedFieldsFromFile : boost::noncopyable
{
public:
    using BT = typename Metadata<N>::BT;
    using BlockId = typename BT::BlockId;
    using BlockIndex = typename BT::BlockIndex;
    using vector_type = typename BlockTree<N>::vector_type;
    using NodeIndex = typename BlockTreeNodes<N, typename Metadata<N>::BT>::NodeIndex;

    BlockTreeIMappedFieldsFromFile(
            const Metadata<N>& metadata,
            MeshProvider& meshProvider) :
        m_metadata(metadata),
        m_meshProvider(meshProvider),
        m_allFieldIndices(m_meshProvider.fieldVariables())
    {
    }

    std::vector<std::string> fieldNames() const
    {
        auto allNames = m_meshProvider.variables();
        std::vector<std::string> result;
        for (auto fieldIndex : m_allFieldIndices)
            result.push_back(allNames[fieldIndex]);
        return result;
    }

private:
    template<MeshElementType elementType>
    void generateZone(
            std::vector<std::vector<real_type>>& fieldValues,
            std::vector<std::vector<real_type>>& weightValues,
            const typename MeshProvider::Zone& zone,
            std::istream& fieldMap) const
    {
        auto fieldCount = fieldValues.size();
        BOOST_ASSERT(weightValues.size() == fieldCount);
        std::vector<real_type> buf(fieldCount);
        BinaryReader reader(fieldMap);
        auto elements = zone.elements();
        while (true) {
            auto ielement = reader.read<unsigned int>();
            if (ielement == ~0u)
                break;
            auto nodeNumber = reader.read<unsigned int>();
            auto param = reader.read<Vec<N,real_type>>();
            auto elementIter = elements.begin() + ielement;
            auto elementNodes = elementIter.elementNodes();

            ElementApprox<N, elementType>::approx(buf.data(), elementNodes, param);
            for (auto ifield=0u; ifield<fieldCount; ++ifield) {
                fieldValues[ifield][nodeNumber] = buf[ifield];
                weightValues[ifield][nodeNumber] = make_real(1);
            }
        }
    }

public:

    void generate(
            std::vector<std::vector<real_type>>& fieldValues,
            std::vector<std::vector<real_type>>& weightValues,
            const std::vector<unsigned int>& fieldIndices,
            real_type noFieldValue,
            std::istream& fieldMap) const
    {
        using namespace BlockTreeIMappedFieldsFromFile_detail;
        auto fieldCount = fieldValues.size();
        BOOST_ASSERT(weightValues.size() == fieldCount);
        BOOST_ASSERT(fieldIndices.size() == fieldCount);
        std::vector<unsigned int> vars(fieldCount);
        for (auto i=0; i<fieldCount; ++i)
            vars[i] = m_allFieldIndices[fieldIndices[i]];
        for (auto zone : m_meshProvider.zones(cache(), vars)) {
            ZoneGeneratorDispatch<N>::template dispatch<MeshProvider>({
                m_metadata, fieldValues, weightValues, zone, fieldMap
            });
        }

        const auto Tol = make_real(1e-5);
        for(auto ifield=0u; ifield<fieldCount; ++ifield) {
            auto& fv = fieldValues[ifield];
            auto& fw = weightValues[ifield];
            std::transform(
                        fv.begin(), fv.end(), fw.begin(), fv.begin(),
                        [Tol, noFieldValue](real_type field, real_type weight)
            {
                return weight < Tol? noFieldValue: field / weight;
            });
        }
    }

    void map(
            real_type *weightValues,
            const BlockId& subtreeRoot,
            const BoundingCube<N, real_type>& subtreePos,
            const BlockTreeNodes<N, BT>& subtreeNodes,
            unsigned int processedChildSubtrees,
            std::ostream& fieldMap) const
    {
        using namespace BlockTreeIMappedFieldsFromFile_detail;
        std::vector<BoundingCube<N, real_type>> processedAreas;
        {
            BlockIndex childIndex;
            auto iChildSubtree = 0u;
            auto childSize = make_real(0.5)*subtreePos.size();
            do {
                if (processedChildSubtrees & (1 << iChildSubtree)) {
                    auto childMinPos = subtreePos.min() + ScalarOrMultiIndex<N,real_type>::fromMultiIndex(childIndex*childSize);
                    processedAreas.push_back(BoundingCube<N, real_type>(childMinPos, childSize));
                }
                ++iChildSubtree;
            }
            while (inc01MultiIndex(childIndex));
        }

        for (auto zone : m_meshProvider.zones(cache(), coordVars())) {
            ZoneMapperDispatch<N>::template dispatch<MeshProvider>({
                m_metadata, weightValues, subtreeRoot, subtreePos, subtreeNodes, zone, processedAreas, fieldMap
            });
        }
    }

private:
    const Metadata<N>& m_metadata;
    MeshProvider& m_meshProvider;
    std::vector<unsigned int> m_allFieldIndices;

    mutable typename MeshProvider::Cache m_cache;
    typename MeshProvider::Cache cache() const
    {
        if (!m_cache)
            m_cache = m_meshProvider.makeCache();
        return m_cache;
    }

    mutable std::vector<unsigned int> m_coordVars;
    const std::vector<unsigned int>& coordVars() const
    {
        if (m_coordVars.empty()) {
            m_coordVars = m_meshProvider.coordinateVariables();
            if (m_coordVars.size() != N)
                throw std::runtime_error("Invalid dimension in the mesh file");
        }
        return m_coordVars;
    }
};

} // s3dmm
