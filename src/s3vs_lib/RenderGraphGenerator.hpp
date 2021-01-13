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

#include "silver_bullets/task_engine.hpp"

namespace s3vs
{

namespace GraphTask
{
enum
{
    RenderStateInit,
    RenderStateUpdate,
    Render,
    ClearTimestamps,
    EnableNvProf,
    Assemble,
    FinalCompose
};
} // namespace GraphTask

enum
{
    RenderResourceId,
    AssembleResourceId
};

constexpr unsigned int taskResourceIdOf(
    unsigned int node, unsigned int resourceId)
{
    return (node << 1) | resourceId;
}

constexpr unsigned int nodeIdOf(unsigned int taskResourceId)
{
    return taskResourceId >> 1;
}

constexpr unsigned int resourceIdOf(unsigned int taskResourceId)
{
    return taskResourceId & 1;
}

class RenderGraphGenerator
{
public:
    using TG = silver_bullets::task_engine::TaskGraph;
    void setAffinity(const std::vector<size_t>& affinity)
    {
        m_valid &= affinity == m_affinity;
        m_affinity = affinity;
    }
    TG getRenderGraph() const
    {
        updateGraph();
        return m_graph;
    }
    std::vector<size_t> getWorkerTasks() const
    {
        updateGraph();
        return m_workerTasks;
    }
    size_t getComposeTask() const
    {
        updateGraph();
        return m_composeTask;
    }

private:
    mutable bool m_valid = false;
    mutable TG m_graph;
    mutable std::vector<size_t> m_workerTasks;
    mutable size_t m_composeTask = ~0;
    std::vector<size_t> m_affinity;

    void updateGraph() const;
};

} // namespace s3vs
