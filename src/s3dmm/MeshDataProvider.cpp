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

#include "s3dmm/MeshDataProvider.hpp"
#include "s3dmm/mesh_reader.hpp"
#include "foreach_byindex32.hpp"

#include <filesystem>
#include <numeric>

namespace s3dmm {

namespace {

inline std::size_t flattenIndex(const unsigned int *ivec, const unsigned int *size, unsigned int dim)
{
    switch (dim)
    {
    case 1:
        return ivec[0];
    case 2:
        return ivec[0] + size[0]*ivec[1];
    case 3:
        return ivec[0] + size[0]*(ivec[1] + size[1]*ivec[2]);
    default:
        BOOST_ASSERT(false);
        throw std::runtime_error("Invalid grid dimension");
    }
}

inline void vectorizeIndex(unsigned int *ivec, std::size_t iflat, const unsigned int *size, unsigned int dim)
{
    switch (dim)
    {
    case 1:
        ivec[0] = static_cast<unsigned int>(iflat);
        break;
    case 2:
        ivec[0] = static_cast<unsigned int>(iflat % size[0]);
        iflat /= size[0];
        ivec[1] = static_cast<unsigned int>(iflat);
        break;
    case 3:
        ivec[0] = static_cast<unsigned int>(iflat % size[0]);
        iflat /= size[0];
        ivec[1] = static_cast<unsigned int>(iflat % size[1]);
        iflat /= size[1];
        ivec[2] = static_cast<unsigned int>(iflat);
        break;
    default:
        BOOST_ASSERT(false);
        throw std::runtime_error("Invalid grid dimension");
    }
}


} // anonymous namespace

class MeshDataProvider::ZonesCache : boost::noncopyable
{
public:
    std::vector<unsigned int> variables;
    std::vector<real_type> nodeData;
    std::vector<unsigned int> elementData;
    unsigned int zoneIndex = ~0u;
    unsigned int nodesPerElement = 0;
    unsigned int gridDim = 0u;
    unsigned int structuredGridSize[3] = {0u, 0u, 0u};  // Number of elements (not nodes!) in the x, y, and z dimensions
    bool structuredGrid = false;
};

class MeshDataProvider::Impl
{
public:
    Impl() = default;
    Impl(const std::string& fileName, MeshReaderType meshReaderType) :
        m_meshReaderType(meshReaderType)
    {
        open(fileName);
    }
    ~Impl() {
        close();
    }

    void open(const std::string& fileName)
    {
        auto rd = reader();
        rd->Open(fileName);
        m_tecplotHeader = rd->GetHeader();
        m_tecplotZoneInfo = rd->GetZoneInfo();
    }

    void close()
    {
        reader()->Close();
        m_tecplotHeader = tecplot_rw::TecplotHeader();
    }

    const std::vector<std::string>& variables() const {
        return m_tecplotHeader.vars;
    }

    Zones zones(const Cache& cache) const
    {
        auto& cd = *cache.get();
        cd.variables.resize(m_tecplotHeader.vars.size());
        std::iota(cd.variables.begin(), cd.variables.end(), 0u);
        return Zones(
                    ZonesIterator(this, &cd, 0u),
                    ZonesIterator(this, &cd, m_tecplotZoneInfo.size()));
    }

    Zones zones(const Cache& cache, range<const unsigned int*> vars) const
    {
        auto& cd = *cache.get();
        if (cd.variables != vars) {
            cd.variables.resize(vars.end() - vars.begin());
            std::copy(vars.begin(), vars.end(), cd.variables.begin());
            cd.zoneIndex = ~0u;
        }
        return Zones(
                    ZonesIterator(this, &cd, 0u),
                    ZonesIterator(this, &cd, m_tecplotZoneInfo.size()));
    }

    unsigned int zoneCount() const {
        return static_cast<unsigned int>(m_tecplotZoneInfo.size());
    }

    ZonesCache *newZonesCache() const {
        return new ZonesCache;
    }
    void deleteZonesCache(const ZonesCache *cache) const {
        delete cache;
    }


    const tecplot_rw::ZoneInfo& tecplotZoneInfo(unsigned int index) const {
        return m_tecplotZoneInfo.at(index);
    }

    void readZone(ZonesCache *cd, unsigned int index) const
    {
        if (cd->zoneIndex != index) {
            auto& zi = m_tecplotZoneInfo.at(index);
            auto rd = reader();
            std::vector<real_type> buf(zi.uBufSizeVar);
            rd->GetZoneVariableValues(index, buf.data());
            auto dstVarCount = static_cast<unsigned int>(cd->variables.size());
            cd->nodeData.resize(zi.uNumNode*dstVarCount);
            auto srcVarCount = m_tecplotHeader.vars.size();
            auto src = buf.data();
            auto dst = cd->nodeData.data();
            for (auto inode=0u; inode<zi.uNumNode; ++inode, src+=srcVarCount, dst+=dstVarCount)
                for (auto ivar=0u; ivar<dstVarCount; ++ivar)
                    dst[ivar] = make_real(src[cd->variables[ivar]]);
            buf.clear();
            buf.shrink_to_fit();

            cd->elementData.resize(zi.uBufSizeCnt);
            if (!cd->elementData.empty())
                rd->GetZoneConnectivity(index, cd->elementData.data());

            switch (zi.uElemType) {
            case tecplot_rw::TECPLOT_ET_TRI:
            case tecplot_rw::TECPLOT_ET_QUAD:
                cd->gridDim = 2u;
                break;
            case tecplot_rw::TECPLOT_ET_TET:
            case tecplot_rw::TECPLOT_ET_BRICK:
                cd->gridDim = 3u;
                break;
            default:
                throw std::runtime_error("Unknown element type");
            }

            cd->zoneIndex = index;
            cd->nodesPerElement = static_cast<unsigned int>(tecplot_rw::NodePerElem(zi.uElemType));
            cd->structuredGrid = zi.bIsOrdered;
            if (zi.bIsOrdered) {
                for (auto d=0u; d<3; ++d) {
                    if (zi.ijk[d] == 0)
                        throw std::runtime_error("Zero ijk is found in tecplot file");
                    cd->structuredGridSize[d] = static_cast<unsigned int>(zi.ijk[d]) - 1u;
                }
            }
            else
                std::fill(cd->structuredGridSize, cd->structuredGridSize+3, 0u);
        }
    }

private:
    MeshReaderType m_meshReaderType;
    mutable tecplot_rw::ITecplotReaderPtr m_reader;
    tecplot_rw::ITecplotReader *reader() const {
        if (!m_reader) {
            switch (m_meshReaderType) {
            case MeshReaderType::Tecplot:
                m_reader = tecplot_rw::CreateReader();
                break;
            case MeshReaderType::Binary:
                m_reader = CreateBinaryMeshReader();
                break;
            case MeshReaderType::Tsagi:
                m_reader = CreateTsagiMeshReader();
                break;
            case MeshReaderType::Imamod_g:
                m_reader = CreateImamod_gMeshReader();
                break;
            }
        }
        return m_reader.get();
    }
    tecplot_rw::TecplotHeader m_tecplotHeader;
    tecplot_rw::ZoneInfo_v m_tecplotZoneInfo;
};



auto MeshDataProvider::MeshNodesIterator::dereference() const -> const NodeData&
{
    auto d = m_cache->nodeData.data() + m_nodeDataOffset;
    return m_nodeData = { d, d+m_cache->variables.size() };
}

void MeshDataProvider::MeshNodesIterator::increment()
{
    m_nodeDataOffset += m_cache->variables.size();
}

bool MeshDataProvider::MeshNodesIterator::equal(const MeshNodesIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache);
    return m_nodeDataOffset == that.m_nodeDataOffset;
}

void MeshDataProvider::MeshNodesIterator::advance(std::ptrdiff_t distance)
{
    m_nodeDataOffset += static_cast<std::ptrdiff_t>(m_cache->variables.size()) * distance;
}

std::ptrdiff_t MeshDataProvider::MeshNodesIterator::distance_to(const MeshNodesIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache);
    auto d = static_cast<std::ptrdiff_t>(that.m_nodeDataOffset) - static_cast<std::ptrdiff_t>(m_nodeDataOffset);
    auto vsize = static_cast<std::ptrdiff_t>(m_cache->variables.size());
    BOOST_ASSERT(std::abs(d) % vsize == 0);
    return d / vsize;
}



auto MeshDataProvider::MeshElementsIterator::dereference() const -> const ElementData&
{
    if (m_cache->structuredGrid) {
        if (m_cache->gridDim == 2u) {
            auto sy = m_cache->structuredGridSize[0]+1;
            auto n0 = m_elementIndex[0] + sy*m_elementIndex[1];
            m_buf[0] = n0;
            m_buf[1] = n0+1;
            m_buf[2] = n0+sy+1;
            m_buf[3] = n0+sy;
        }
        else {
            BOOST_ASSERT(m_cache->gridDim == 3u);
            auto sy = m_cache->structuredGridSize[0]+1;
            auto sz = (m_cache->structuredGridSize[1]+1) * sy;
            auto n0 = m_elementIndex[0] + sy*m_elementIndex[1] + sz*m_elementIndex[2];
            m_buf[0] = n0;
            m_buf[1] = n0+1;
            m_buf[2] = n0+sy+1;
            m_buf[3] = n0+sy;
            m_buf[4] = n0+sz;
            m_buf[5] = n0+sz+1;
            m_buf[6] = n0+sz+sy+1;
            m_buf[7] = n0+sz+sy;
        }
        return m_elementData = { m_buf, m_buf+m_cache->nodesPerElement };
    }
    else {
        auto d = m_cache->elementData.data() + m_elementDataOffset;
        return m_elementData = { d, d+m_cache->nodesPerElement };
    }
}

void MeshDataProvider::MeshElementsIterator::increment()
{
    if (m_cache->structuredGrid) {
        for (auto d=0u; d<m_cache->gridDim; ++d) {
            if (m_elementIndex[d]+1 < m_cache->structuredGridSize[d]) {
                ++m_elementIndex[d];
                for (auto d2=0u; d2<d; ++d2)
                    m_elementIndex[d2] = 0u;
                return;
            }
        }
        std::copy(m_cache->structuredGridSize, m_cache->structuredGridSize+3, m_elementIndex);
    }
    else
        m_elementDataOffset += m_cache->nodesPerElement;
}

bool MeshDataProvider::MeshElementsIterator::equal(const MeshElementsIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache);
    if (m_cache->structuredGrid)
        return std::equal(m_elementIndex, m_elementIndex+3, that.m_elementIndex);
    else
        return m_elementDataOffset == that.m_elementDataOffset;
}

void MeshDataProvider::MeshElementsIterator::advance(std::ptrdiff_t distance)
{
    if (m_cache->structuredGrid) {
        vectorizeIndex(
                    m_elementIndex,
                    flattenIndex(m_elementIndex, m_cache->structuredGridSize, m_cache->gridDim) + distance,
                    m_cache->structuredGridSize, m_cache->gridDim);
    }
    else
        m_elementDataOffset += static_cast<std::ptrdiff_t>(m_cache->nodesPerElement)*distance;
}

std::ptrdiff_t MeshDataProvider::MeshElementsIterator::distance_to(const MeshElementsIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache);
    if (m_cache->structuredGrid) {
        return
            static_cast<ptrdiff_t>(flattenIndex(that.m_elementIndex, m_cache->structuredGridSize, m_cache->gridDim)) -
            static_cast<ptrdiff_t>(flattenIndex(m_elementIndex, m_cache->structuredGridSize, m_cache->gridDim));
    }
    else
        return static_cast<std::ptrdiff_t>(that.m_elementDataOffset) - static_cast<std::ptrdiff_t>(m_elementDataOffset);
}



auto MeshDataProvider::ElementNodesIterator::dereference() const -> const NodeData&
{
    auto nodeNumber = *(m_elementData.begin() + m_localNodeIndex);
    if (nodeNumber == ~0u) {
        BOOST_ASSERT(m_firstMissingNodeIndex <= m_localNodeIndex);
        if (m_cache->gridDim == 2u) {
            BOOST_ASSERT(m_localNodeIndex == 3);
            BOOST_ASSERT(m_cache->nodesPerElement == 4u);
            nodeNumber = *(m_elementData.begin());
        }
        else {
            BOOST_ASSERT(m_cache->gridDim == 3u);
            BOOST_ASSERT(m_cache->nodesPerElement == 8u);
            unsigned int localMergedNodeIndex;
            switch (m_firstMissingNodeIndex) {
                case 4: // Tetrahedron
                    localMergedNodeIndex = m_localNodeIndex == 4? 0: 3;
                    break;
                case 5: // Pyramid
                    localMergedNodeIndex = 4;
                    break;
                case 6: // Prism
                    localMergedNodeIndex = m_localNodeIndex == 6? 0: 3;
                    break;
                default:
                    BOOST_ASSERT(false);
                    localMergedNodeIndex = ~0u;
            }
            nodeNumber = *(m_elementData.begin() + localMergedNodeIndex);
        }
    }
    else {
        BOOST_ASSERT(m_firstMissingNodeIndex == ~0u);
    }
    auto d = m_cache->nodeData.data() + nodeNumber * m_cache->variables.size();
    return m_nodeData = { d, d+m_cache->variables.size() };
}

void MeshDataProvider::ElementNodesIterator::increment()
{
    ++m_localNodeIndex;
    if (m_firstMissingNodeIndex == ~0u &&
        m_localNodeIndex < m_cache->nodesPerElement &&
        *(m_elementData.begin() + m_localNodeIndex) == ~0u)
    {
        m_firstMissingNodeIndex = m_localNodeIndex;
    }
}

bool MeshDataProvider::ElementNodesIterator::equal(const ElementNodesIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache && m_elementData == that.m_elementData);
    return m_localNodeIndex == that.m_localNodeIndex;
}



MeshElementType MeshDataProvider::Zone::elementType() const
{
    auto& zi = m_impl->tecplotZoneInfo(m_zoneIndex);
    return static_cast<MeshElementType>(zi.uElemType);  // TODO better
}

unsigned int MeshDataProvider::Zone::nodesPerElement() const
{
    m_impl->tecplotZoneInfo(m_zoneIndex);
    return m_cache->nodesPerElement;
}

auto MeshDataProvider::Zone::nodes() const -> MeshNodes
{
    return MeshNodes(
                MeshNodesIterator(m_impl, m_cache, 0),
                MeshNodesIterator(m_impl, m_cache, static_cast<unsigned int>(m_cache->nodeData.size())));
}

unsigned int MeshDataProvider::Zone::nodeCount() const {
    return static_cast<unsigned int>(m_cache->nodeData.size() / m_cache->variables.size());
}

auto MeshDataProvider::Zone::elements() const -> MeshElements
{
    if (m_cache->structuredGrid) {
        unsigned int i0[3] = {0u, 0u, 0u};
        return MeshElements(
                    MeshElementsIterator(m_impl, m_cache, i0),
                    MeshElementsIterator(m_impl, m_cache, m_cache->structuredGridSize));
    }
    else
        return MeshElements(
                    MeshElementsIterator(m_impl, m_cache, 0u),
                    MeshElementsIterator(m_impl, m_cache, static_cast<unsigned int>(m_cache->elementData.size())));
}

unsigned int MeshDataProvider::Zone::elementCount() const
{
    if (m_cache->structuredGrid) {
        auto result = 1u;
        for (auto d=0u; d<m_cache->gridDim; ++d)
            result *= m_cache->structuredGridSize[d];
        return result;
    }
    else
        return static_cast<unsigned int>(m_cache->elementData.size() / m_cache->nodesPerElement);
}

std::vector<unsigned int> MeshDataProvider::Zone::structuredMeshShape() const
{
    std::vector<unsigned int> result;
    if (m_cache->structuredGrid) {
        for (auto d=0u; d<m_cache->gridDim; ++d)
            result.push_back(m_cache->structuredGridSize[d]+1);
    }
    return result;
}




auto MeshDataProvider::ZonesIterator::dereference() const -> const Zone&
{
    m_impl->readZone(m_cache, m_zoneIndex);
    return m_zone = Zone(m_impl, m_cache, m_zoneIndex);
}

void MeshDataProvider::ZonesIterator::increment()
{
    ++m_zoneIndex;
}

bool MeshDataProvider::ZonesIterator::equal(const ZonesIterator& that) const
{
    BOOST_ASSERT(m_impl == that.m_impl && m_cache == that.m_cache);
    return m_zoneIndex == that.m_zoneIndex;
}



MeshReaderType MeshDataProvider::guessMeshReaderType(const std::string& fileName)
{
    auto ext = std::filesystem::path(fileName).extension();
    if (ext == ".tec")
        return MeshReaderType::Tecplot;
    else if (ext == ".bin")
        return MeshReaderType::Binary;
    else if (ext == ".anim")
        return MeshReaderType::Tsagi;
    else if (ext == ".imm-g")
        return MeshReaderType::Imamod_g;
    else
        throw std::invalid_argument("Failed to determine file type by name - please specify .tec, .bin, or .anim filename extension");
}

MeshDataProvider::MeshDataProvider() :
    m_impl(std::make_unique<Impl>()) {}

MeshDataProvider::MeshDataProvider(const std::string& fileName) :
    m_impl(std::make_unique<Impl>(fileName, guessMeshReaderType(fileName)))
{
}

MeshDataProvider::MeshDataProvider(const std::string& fileName, MeshReaderType meshReaderType) :
    m_impl(std::make_unique<Impl>(fileName, meshReaderType))
{
}

MeshDataProvider::~MeshDataProvider() = default;

void MeshDataProvider::open(const std::string& fileName) {
    m_impl->open(fileName);
}

void MeshDataProvider::close() {
    m_impl->close();
}

const std::vector<std::string>& MeshDataProvider::variables() const {
    return m_impl->variables();
}

std::vector<unsigned int> MeshDataProvider::coordinateVariables() const
{
    auto vars = variables();
    unsigned int xyz[3] = {~0u, ~0u, ~0u};
    char coordinateNames[3] = {'x', 'y', 'z'};
    foreach_byindex32(i, vars) {
        auto name = vars[i];
        for (auto ic=0; ic<3; ++ic) {
            if (name.size() == 1) {
                auto nameChar = name[0];
                if (nameChar == coordinateNames[ic] || nameChar == toupper(coordinateNames[ic])) {
                    if (xyz[ic] == ~0u)
                        xyz[ic] = i;
                    else
                        throw std::runtime_error("Failed to extract coordinate indices: same name found more than once");
                }
            }
        }
    }
    auto dim = 0;
    for (; dim<3; ++dim)
        if (xyz[dim] == ~0u)
            break;
    for (auto i=dim; i<3; ++i) {
        if (xyz[i] != ~0u)
            throw std::runtime_error("Failed to extract coordinate indices: [y], [z], [x,z], and [y,z] are all not accepted");
    }
    BOOST_ASSERT(dim <= 3);
    return std::vector<unsigned int>(xyz, xyz+dim);
}

std::vector<unsigned int> MeshDataProvider::fieldVariables() const
{
    auto cvars = coordinateVariables();
    std::vector<unsigned int> result(variables().size());
    iota(result.begin(), result.end(), 0u);
    auto n = result.size();
    for (unsigned int i=0; i<n; ++i) {
        if (std::find(cvars.begin(), cvars.end(), result[i]) != cvars.end()) {
            result.erase(result.begin() + i);
            --n;
            --i;
        }
    }
    return result;
}

auto MeshDataProvider::zones(const Cache& cache) const -> Zones {
    return m_impl->zones(cache);
}

auto MeshDataProvider::zones(const Cache& cache, range<const unsigned int*> vars) const -> Zones {
    return m_impl->zones(cache, vars);
}

unsigned int MeshDataProvider::zoneCount() const
{
    return m_impl->zoneCount();
}

auto MeshDataProvider::newZonesCache() const -> ZonesCache* {
    return m_impl->newZonesCache();
}

void MeshDataProvider::deleteZonesCache(const ZonesCache* cache) const {
    return m_impl->deleteZonesCache(cache);
}

} // s3dmm
