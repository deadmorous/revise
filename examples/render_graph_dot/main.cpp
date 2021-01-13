#include <exception>
#include <iostream>
#include <sstream>

#include "silver_bullets/iterate_struct/json_doc_converter.hpp"
#include "silver_bullets/iterate_struct/json_doc_io.hpp"

#include <boost/program_options.hpp>

#include "RenderGraphGenerator.hpp"

using namespace std;
using namespace silver_bullets::task_engine;

class RenderGraph2Dot
{
public:
    RenderGraph2Dot(const TaskGraph& graph) : m_graph(graph)
    {
        ostringstream os;
        os << "digraph G {" << endl;
        for (size_t i = 0; i < m_graph.taskInfo.size(); i++)
            addTask(m_graph.taskInfo[i].task);

        for (const auto& task: m_taskPriv)
            os << task.taskString() << endl;

        for (const auto& cnt: m_graph.connections)
            os << connectionString(cnt) << endl;

        os << "{ rank=same; ";
        for (const auto& task: m_taskPriv)
            if (task.task.taskFuncId == s3vs::GraphTask::Render)
                os << task.id << " ";
        os << "}" << endl;

        os << "}" << endl;
        m_result = os.str();
    }
    operator const string&() const
    {
        return m_result;
    }

private:
    const TaskGraph& m_graph;
    string m_result;

    class VertexCounter
    {
    public:
        VertexCounter(const string& prefix) : m_prefix(prefix)
        {
        }
        string vertexName() const
        {
            return m_prefix + std::to_string(m_count);
        }
        string vertexNameNext() const
        {
            auto s = vertexName();
            ++m_count;
            return s;
        }

    private:
        string m_prefix;
        mutable unsigned int m_count{0};
    };
    const std::array<VertexCounter, s3vs::GraphTask::FinalCompose + 1>
        m_counters = {string("RenderStateInit"),
                      string("RenderStateUpdate"),
                      string("Render"),
                      string("ClearTimestamps"),
                      string("EnableNvProf"),
                      string("Assemble"),
                      string("FinalCompose")};

    struct TaskPriv
    {
        Task task;
        string id;
        string label;
        string taskString() const
        {
            ostringstream os;
            os << id << " ["
               << "label=\"" << label << "\"";
            if (task.taskFuncId == s3vs::GraphTask::Render)
                os << " shape=box";
            os << "]";
            return os.str();
        }
    };
    vector<TaskPriv> m_taskPriv;

    void addTask(const Task& task)
    {
        TaskPriv t;
        t.task = task;
        BOOST_ASSERT(task.taskFuncId <= s3vs::GraphTask::FinalCompose);
        auto& counter = m_counters[task.taskFuncId];
        t.id = counter.vertexNameNext();
        t.label = t.id + "\nAt node "
                  + std::to_string(s3vs::nodeIdOf(task.resourceType));
        m_taskPriv.push_back(t);
    }

    string connectionString(const Connection& c) const
    {
        ostringstream os;
        os << m_taskPriv[c.from.taskId].id << "->"
           << m_taskPriv[c.to.taskId].id;
        return os.str();
    }
};

string renderGraph2Dot(const TaskGraph& graph)
{
    return RenderGraph2Dot(graph);
}

struct AppOptions
{
    string output;
    vector<size_t> affinity;

    static AppOptions exampleVal()
    {
        AppOptions opts{"render_graph_dot.gv", {4, 4, 2}};
        return opts;
    }
};

SILVER_BULLETS_DESCRIBE_STRUCTURE_FIELDS(AppOptions, output, affinity);

void run(int argc, char* argv[])
{
    using namespace silver_bullets;
    namespace po = boost::program_options;
    po::options_description desc;
    desc.add_options()(
        "config,c",
        po::value<std::string>(),
        "Path to JSON config")("example,e", "Show example config and exit")(
        "out_cfg,o",
        po::value<string>(),
        "Path to file where example config is to be saved");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.size() == 0)
    {
        cout << desc << endl;
        return;
    }

    AppOptions opts;

    if (vm.count("example"))
    {
        opts = AppOptions::exampleVal();
        auto json = iterate_struct::to_json_doc(opts);
        if (vm.count("out_cfg"))
        {
            auto name = vm["out_cfg"].as<string>();
            iterate_struct::write_json_doc(name, json);
        }
        else
        {
            iterate_struct::write_json_doc(cout, json);
            cout << endl;
        }
        return;
    }

    auto cfgName = vm["config"].as<string>();
    auto json = iterate_struct::read_json_doc(cfgName);

    opts = iterate_struct::from_json_doc<decltype(opts)>(json);

    s3vs::RenderGraphGenerator rgg;
    rgg.setAffinity(opts.affinity);
    auto graph = rgg.getRenderGraph();
    auto dotScript = renderGraph2Dot(graph);

    if (opts.output.empty())
        cout << dotScript << endl;
    else
    {
        ofstream f(opts.output);
        if (!f.is_open())
            throw runtime_error(
                "Failed to open output file '" + opts.output + "'");
        f << dotScript << endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        run(argc, argv);
        return EXIT_SUCCESS;
    }
    catch (exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
