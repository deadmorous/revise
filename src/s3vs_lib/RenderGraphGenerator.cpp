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


#include "RenderGraphGenerator.hpp"

namespace s3vs
{

namespace
{

struct NodeData
{
    std::vector<size_t> workers;
    size_t assembly = ~0;
};

} // namespace

void RenderGraphGenerator::updateGraph() const
{
    if (m_valid)
        return;
    m_valid = true;

    m_workerTasks.clear();

    silver_bullets::task_engine::TaskGraphBuilder builder;

    std::map<size_t, size_t> task2node;

    auto addTask = [&](size_t inputCount,
                       size_t outputCount,
                       auto taskFuncId,
                       auto resourceId) {
        auto task =
            builder.addTask(inputCount, outputCount, taskFuncId, resourceId);
        BOOST_ASSERT(task2node.find(task) == task2node.end());
        task2node[task] = nodeIdOf(resourceId);
        return task;
    };

    auto nodeOfTask = [&](size_t task) {
        auto it = task2node.find(task);
        BOOST_ASSERT(it != task2node.end());
        return it->second;
    };

    using V = std::vector<size_t>;
    std::function<V(V)> addAssemble = [&](V prevLayer) {
        if (prevLayer.size() < 2)
            return prevLayer;
        V newLayer;
        for (size_t i = 1; i < prevLayer.size(); i += 2)
        {
            auto node = nodeOfTask(prevLayer[i - 1]);
            auto assemble = addTask(
                3,
                1,
                GraphTask::Assemble,
                taskResourceIdOf(node, AssembleResourceId));
            builder.connect(prevLayer[i - 1], 0, assemble, 0);
            builder.connect(prevLayer[i], 0, assemble, 1);
            newLayer.push_back(assemble);
        }
        if (prevLayer.size() % 2)
            newLayer.push_back(prevLayer.back());
        return addAssemble(newLayer);
    };

    std::vector<NodeData> nodeData;
    for (size_t node = 0; node < m_affinity.size(); node++)
    {
        NodeData nd;
        for (size_t i = 0; i < m_affinity[node]; i++)
        {
            auto render = addTask(
                2,
                1,
                GraphTask::Render,
                taskResourceIdOf(node, RenderResourceId));
            nd.workers.push_back(render);
        }
        auto topAsm = addAssemble(nd.workers);
        if (topAsm.size())
            nd.assembly = topAsm[0];
        if (!nd.workers.empty())
            nodeData.push_back(nd);
        m_workerTasks.insert(
            m_workerTasks.end(), nd.workers.begin(), nd.workers.end());
    }

    std::vector<size_t> asms;
    for (const auto& x: nodeData)
        asms.push_back(x.assembly);

    auto top = addAssemble(asms);

    m_composeTask = ~0;
    if (top.size())
    {
        auto node = nodeOfTask(top[0]);
        m_composeTask = addTask(
            3,
            1,
            GraphTask::FinalCompose,
            taskResourceIdOf(node, AssembleResourceId));
        builder.connect(top[0], 0, m_composeTask, 0);
    }

    m_graph = builder.taskGraph();
}

} // namespace s3vs
