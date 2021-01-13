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

#include "RenderGraphGenerator.hpp"

namespace s3vs
{

class RenderGraphGeneratorPool
{
public:
    using RGG = RenderGraphGenerator;
    void init(unsigned int nodeCount, unsigned int threadPerNode)
    {
        m_nodeCount = nodeCount;
        m_threadPerNode = threadPerNode;
        m_generators.resize(nodeCount * threadPerNode);
        for (size_t i = 0; i < m_generators.size(); i++)
            m_generators[i].setAffinity(getAffinity(i + 1));
    }
    const RGG& getGenerator(unsigned int threadCount) const
    {
        if (threadCount > m_generators.size())
            throw std::runtime_error(
                "RenderGraphGeneratorPool: threadCount is out of range");
        if (threadCount == 0)
            throw std::runtime_error(
                "RenderGraphGeneratorPool: threadCount must be positive");
        return m_generators[threadCount - 1];
    }

private:
    std::vector<RGG> m_generators;
    unsigned int m_nodeCount{1};
    unsigned int m_threadPerNode{1};

    std::vector<size_t> getAffinity(unsigned int threadCount) const
    {
        BOOST_ASSERT(threadCount > 0 && threadCount <= m_generators.size());
        unsigned int nodeCountFull = threadCount / m_threadPerNode;
        unsigned int remained = threadCount % m_threadPerNode;
        std::vector<size_t> affinity(nodeCountFull, m_threadPerNode);
        if (remained)
            affinity.push_back(remained);
        return affinity;
    }
};

} // namespace s3vs
