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


#include <list>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>

#include <boost/assert.hpp>

template<class T>
class SingleElementAllocator
{
public:
    using value_type = T;
    SingleElementAllocator() {
        m_chunks.emplace_back(m_chunkSize);
    }

    explicit SingleElementAllocator(std::size_t chunkSize) : m_chunkSize(chunkSize) {
        m_chunks.emplace_back(m_chunkSize);
    }

    std::size_t chunkSize() const {
        return m_chunkSize;
    }

    template<class T2>
    SingleElementAllocator(const SingleElementAllocator<T2>& that) :
        m_chunkSize(that.chunkSize())
    {
        m_chunks.emplace_back(m_chunkSize);
    }

    T *allocate(std::size_t n)
    {
        BOOST_ASSERT(n == 1);
        if (n != 1)
            throw std::invalid_argument("SingleElementAllocator is only suitable for allocating single elements");
        return newNode();
    }

    void deallocate(T* p, std::size_t n)
    {
        BOOST_ASSERT(n == 1);
        if (n != 1)
            throw std::invalid_argument("SingleElementAllocator is only suitable for deallocating single elements");
        deleteNode(p);
    }

private:
    std::size_t m_chunkSize = 0x100000;
    class Chunk
    {
    public:
        Chunk(std::size_t size) :
            m_memory(size*sizeof(T)),
            m_avail(size),
            m_availSize(size)
        {
            BOOST_ASSERT(size > 0);
            std::iota(m_avail.begin(), m_avail.end(), 0);
        }

        ~Chunk() {
            if (m_availSize != m_avail.size()) {
                BOOST_ASSERT(false);
                std::cerr << "SingleElementAllocator::Chunk::~Chunk(): Destroying allocator before all its memory is deallocated" << std::endl;
                std::terminate();
            }
        }

        T *allocate()
        {
            if (m_availSize == 0)
                return nullptr;
            else {
                auto result = data() + m_avail.at(m_availSize-1);
                --m_availSize;
                return result;
            }
        }

        void deallocate(T *ptr)
        {
            std::size_t index = ptr - data();
            BOOST_ASSERT(index < m_avail.size());
            BOOST_ASSERT(m_availSize < m_avail.size());
            m_avail[m_availSize] = index;
            ++m_availSize;
        }

        bool hasPointer(T* ptr) const
        {
            auto d = data();
            return ptr >= d && ptr < d + m_avail.size();
        }

        void swap(Chunk& that)
        {
            m_memory.swap(that.m_memory);
            m_avail.swap(that.m_avail);
            std::swap(m_availSize, that.m_availSize);
        }

        unsigned int availSize() const {
            return m_availSize;
        }

    private:
        std::vector<char> m_memory;
        std::vector<unsigned int> m_avail;
        unsigned int m_availSize;
        T *data() {
            return reinterpret_cast<T*>(m_memory.data());
        }
        const T *data() const {
            return const_cast<Chunk*>(this)->data();
        }
    };

    std::list<Chunk> m_chunks;

    T *newNode()
    {
        BOOST_ASSERT(!m_chunks.empty());
        auto& chunk = m_chunks.front();
        auto result = chunk.allocate();
        if (!result) {
            m_chunks.emplace_front(m_chunkSize);
            result = m_chunks.front().allocate();
            BOOST_ASSERT(result);
        }
        return result;
    }

    void deleteNode(T *ptr)
    {
        for (auto it=m_chunks.begin(); it!=m_chunks.end(); ++it)
        {
            auto& chunk = *it;
            if (chunk.hasPointer(ptr)) {
                chunk.deallocate(ptr);
                if (it != m_chunks.begin()) {
                    auto& firstChunk = m_chunks.front();
                    if (firstChunk.availSize() < chunk.availSize())
                        chunk.swap(firstChunk);
                }
                return;
            }
        }
        BOOST_ASSERT(false);
        std::cerr << "SingleElementAllocator::deleteNode(): Invalid pointer" << std::endl;
        std::terminate();
    }
};
