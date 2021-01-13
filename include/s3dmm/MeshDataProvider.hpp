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

#include "real_type.hpp"
#include "MeshElementType.hpp"
#include "defs.h"

#include <boost/range/iterator_range.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/noncopyable.hpp>

#include <string>
#include <vector>
#include <memory>

namespace s3dmm {

enum class MeshReaderType
{
    Tecplot,
    Binary,
    Tsagi,
    Imamod_g
};

class S3DMM_API MeshDataProvider : boost::noncopyable
{
private:
    class ZonesCache;
    class Impl;
public:
    class Zone;
    class ElementNodes;
    template<class It>
    using range = boost::iterator_range<It>;

    static MeshReaderType guessMeshReaderType(const std::string& fileName);

    MeshDataProvider();
    explicit MeshDataProvider(const std::string& fileName);
    MeshDataProvider(const std::string& fileName, MeshReaderType meshReaderType);

    ~MeshDataProvider();

    void open(const std::string& fileName);
    void close();

    const std::vector<std::string>& variables() const;
    std::vector<unsigned int> coordinateVariables() const;
    std::vector<unsigned int> fieldVariables() const;

    using NodeData = range<const real_type*>;
    using ElementData = range<const unsigned int*>;

    class MeshNodesIterator : public boost::iterator_facade<
            MeshNodesIterator, const NodeData, boost::random_access_traversal_tag>
    {
    public:
        const NodeData& dereference() const;
        void increment();
        bool equal(const MeshNodesIterator& that) const;
        void advance(std::ptrdiff_t distance);
        std::ptrdiff_t distance_to(const MeshNodesIterator& that) const;

    private:
        const Impl *m_impl = nullptr;
        ZonesCache *m_cache = nullptr;
        unsigned int m_nodeDataOffset = 0u;
        mutable NodeData m_nodeData;
        MeshNodesIterator() = default;
        MeshNodesIterator(const Impl *impl, ZonesCache *cache, unsigned int nodeDataOffset) :
            m_impl(impl),
            m_cache(cache),
            m_nodeDataOffset(nodeDataOffset) {}
        friend class Zone;
    };
    using MeshNodes = range<MeshNodesIterator>;

    class MeshElementsIterator : public boost::iterator_facade<
            MeshElementsIterator, const ElementData, boost::random_access_traversal_tag>
    {
    public:
        const ElementData& dereference() const;
        void increment();
        bool equal(const MeshElementsIterator& that) const;
        void advance(std::ptrdiff_t distance);
        std::ptrdiff_t distance_to(const MeshElementsIterator& that) const;

        ElementNodes elementNodes();

    private:
        const Impl *m_impl = nullptr;
        ZonesCache *m_cache = nullptr;
        unsigned int m_elementDataOffset = 0u;
        unsigned int m_elementIndex[3] = {0u, 0u, 0u};
        constexpr const static unsigned int MaxElementsPerNode = 8;
        mutable unsigned int m_buf[MaxElementsPerNode];
        mutable ElementData m_elementData;
        MeshElementsIterator() = default;
        MeshElementsIterator(const Impl *impl, ZonesCache *cache, unsigned int elementDataOffset) :
            m_impl(impl),
            m_cache(cache),
            m_elementDataOffset(elementDataOffset) {}
        MeshElementsIterator(const Impl *impl, ZonesCache *cache, unsigned int elementIndex[3]) :
            m_impl(impl),
            m_cache(cache)
        {
            std::copy(elementIndex, elementIndex+3, m_elementIndex);
        }
        friend class Zone;
        friend class ElementNodes;
    };
    using MeshElements = range<MeshElementsIterator>;

    class ElementNodesIterator : public boost::iterator_facade<
            ElementNodesIterator, const NodeData, boost::forward_traversal_tag>
    {
    public:
        const NodeData& dereference() const;
        void increment();
        bool equal(const ElementNodesIterator& that) const;

    private:
        const Impl *m_impl = nullptr;
        ZonesCache *m_cache = nullptr;
        ElementData m_elementData;
        unsigned int m_localNodeIndex;
        unsigned int m_firstMissingNodeIndex = ~0;
        mutable NodeData m_nodeData;
        ElementNodesIterator(const Impl *impl, ZonesCache *cache, const ElementData& elementData, unsigned int localNodeIndex) :
            m_impl(impl),
            m_cache(cache),
            m_elementData(elementData),
            m_localNodeIndex(localNodeIndex) {}
        friend class ElementNodes;
    };

    class ElementNodes : public range<ElementNodesIterator>
    {
    public:
        explicit ElementNodes(const MeshElementsIterator& it) :
            range<ElementNodesIterator>(makeRange(it))
        {}
    private:
        static range<ElementNodesIterator> makeRange(const MeshElementsIterator& it)
        {
            auto& elementData = it.dereference();
            return boost::make_iterator_range(
                        ElementNodesIterator(it.m_impl, it.m_cache, elementData, 0u),
                        ElementNodesIterator(it.m_impl, it.m_cache, elementData, static_cast<unsigned int>(elementData.size())));
        }
    };

    class ZonesIterator;

    class Zone
    {
    public:
        MeshElementType elementType() const;
        unsigned int nodesPerElement() const;
        MeshNodes nodes() const;
        unsigned int nodeCount() const;
        MeshElements elements() const;
        unsigned int elementCount() const;
        std::vector<unsigned int> structuredMeshShape() const;

    private:
        const Impl *m_impl = nullptr;
        ZonesCache *m_cache = nullptr;
        unsigned int m_zoneIndex = 0u;
        Zone() = default;
        Zone(const Impl *impl, ZonesCache *cache, unsigned int zoneIndex) :
            m_impl(impl),
            m_cache(cache),
            m_zoneIndex(zoneIndex) {}
        friend class ZonesIterator;
    };

    class ZonesIterator : public boost::iterator_facade<
            ZonesIterator, const Zone, boost::forward_traversal_tag>
    {
    public:
        const Zone& dereference() const;
        void increment();
        bool equal(const ZonesIterator& that) const;

    private:
        const Impl *m_impl;
        ZonesCache *m_cache;
        unsigned int m_zoneIndex;
        ZonesIterator(const Impl *impl, ZonesCache *cache, unsigned int zoneIndex) :
            m_impl(impl),
            m_cache(cache),
            m_zoneIndex(zoneIndex) {}
        mutable Zone m_zone;
        friend class Impl;
    };
    using Zones = range<ZonesIterator>;

    using Cache = std::shared_ptr<ZonesCache>;

    Cache makeCache() const
    {
        return Cache(newZonesCache(), [this](ZonesCache *cache) {
            deleteZonesCache(cache);
        });
    }

    Zones zones(const Cache& cache) const;
    Zones zones(const Cache& cache, range<const unsigned int*> vars) const;

    Zones zones(const Cache& cache, const std::vector<unsigned int>& vars) {
        return zones(cache, boost::make_iterator_range_n(vars.data(), vars.size()));
    }

    template<class It>
    Zones zones(const Cache& cache, It varStart, It varEnd)
    {
        std::vector<unsigned int> vars(std::distance(varStart, varEnd));
        std::copy(varStart, varEnd, vars.begin());
        return zones(cache, vars);
    }
    unsigned int zoneCount() const;

private:
    ZonesCache *newZonesCache() const;
    void deleteZonesCache(const ZonesCache* cache) const;

    std::unique_ptr<Impl> m_impl;
};

inline auto MeshDataProvider::MeshElementsIterator::elementNodes() -> ElementNodes {
    return ElementNodes(*this);
}

} // s3dmm
