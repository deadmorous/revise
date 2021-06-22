#include <iostream>

#include "s3dmm/MeshDataProvider.hpp"
#include "s3dmm/ProgressReport.hpp"

#include "silver_bullets/fs_ns_workaround.hpp"

#include "filename_util.hpp"

#include <boost/program_options.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/transform.hpp>

#include <cgnslib.h>

using namespace std;
using namespace filesystem;
using namespace s3dmm;

template<typename F, typename... Args>
int cgcall(F f, std::function<std::string()> prefixGetter, Args... args)
{
    int id = 0;
    if (f(args..., &id) != CG_OK)
        throw runtime_error(prefixGetter() + ": " + cg_get_error());
    return id;
}

template<typename F, typename... Args>
int cgcall(F f, const char* fname, Args... args)
{
    auto prefixGetter = [fname]()->std::string{ return fname; };
    return cgcall(f, prefixGetter, args...);
}

#define CGCALL(func, ...) cgcall(func, #func, ##__VA_ARGS__)

template<typename T>
constexpr DataType_t cgDataType() noexcept;

template<>
constexpr DataType_t cgDataType<float>() noexcept {
    return RealSingle;
}

template<>
constexpr DataType_t cgDataType<double>() noexcept {
    return RealDouble;
}

ElementType_t cgElementType(MeshElementType t) noexcept
{
    switch (t) {
    case MeshElementType::Triangle:
        return TRI_3;
    case MeshElementType::Quad:
        return QUAD_4;
    case MeshElementType::Tetrahedron:
        return TETRA_4;
    case MeshElementType::Hexahedron:
        return HEXA_8;
    }
}

void writeZoneNodes(int out_file, int out_base, int out_zone, const MeshDataProvider::Zone& zone, size_t dim)
{
    constexpr auto dataType = cgDataType<real_type>();
    auto nodeCount = zone.nodeCount();
    auto nodeRange = zone.nodes();
    for (size_t d=0; d<dim; ++d)
    {
        std::vector<real_type> coord(nodeCount);
        boost::range::transform(nodeRange, coord.begin(), [d](auto& nodeCoords) {return nodeCoords[d]; });
        auto coordName = "Coordinate" + std::string( 1, 'X' + d );
        CGCALL(cg_coord_write, out_file, out_base, out_zone, dataType, coordName.c_str(), coord.data());
    }
}

void writeZoneElements(int out_file, int out_base, int out_zone, const MeshDataProvider::Zone& zone)
{
    auto elementCount = zone.elementCount();
    auto elementType = cgElementType(zone.elementType());
    auto elementRange = zone.elements();
    auto elementNodeCount = elementRange.front().size();
    std::vector<cgsize_t> connectivity(elementCount*elementNodeCount);
    auto ptr = connectivity.data();
    for (auto e : elementRange) {
        BOOST_ASSERT(e.size() == elementNodeCount);
        boost::range::copy(e, ptr);
        ptr += elementNodeCount;
    }
    // Indices in connectivity start with one!
    boost::range::transform(
        connectivity, connectivity.begin(),
        [](auto x) { return x + 1; });
    CGCALL(cg_section_write,
           out_file, out_base, out_zone,
           "Elem", elementType, 1, elementCount, 0, connectivity.data());

}

void writeZoneFields(
    int out_file, int out_base, int out_zone,
    const MeshDataProvider::Zone& zone, const std::vector<unsigned int>& fieldVar,
    const std::vector<std::string>& variableNames)
{
    constexpr auto dataType = cgDataType<real_type>();
    auto out_sol = CGCALL(cg_sol_write, out_file, out_base, out_zone, "Fields", Vertex);

    auto nodeCount = zone.nodeCount();
    auto nodeRange = zone.nodes();

    for (size_t ifield=0, nfields=fieldVar.size(); ifield<nfields; ++ifield )
    {
        std::vector<real_type> field(nodeCount);
        boost::range::transform(
            nodeRange, field.begin(),
            [ifield](auto& nodeCoords)
            { return nodeCoords[ifield]; });
        auto fieldName = variableNames.at(fieldVar[ifield]);
        CGCALL(
            cg_field_write, out_file, out_base, out_zone, out_sol,
            dataType, fieldName.c_str(), field.data());
    }
}

void run(int argc, char* argv[])
{
    std::string inputName;
    std::string outputName;
    auto overwrite = false;

    namespace po = boost::program_options;
    po::options_description desc;
    desc.add_options()
        ("input,i", po::value(&inputName), "Path to input data")
        ("output,o", po::value(&outputName), "Path to CGNS output data")
        ("overwrite", "Overwrite CGNS output if exists");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.find("overwrite") != vm.end())
        overwrite = true;

    if (inputName.empty())
        throw runtime_error("No input mesh has been specified");

    if (!overwrite && exists(outputName))
        throw runtime_error("Output file '" + outputName + "' already exists");
    auto outputDir = path(outputName).parent_path();
    create_directories(outputDir);
    if (!is_directory(outputDir))
        throw runtime_error("Failed to create output dorectory '" + string(outputDir) + "'");

    auto [mainMeshFileName, hasTimeSteps] = firstOutputFrameFileName(inputName);
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Open input and output files");
    MeshDataProvider mainMeshProvider(mainMeshFileName);
    cout << "Opened input file " << mainMeshFileName << endl
         << "Variables: "
         << boost::join(mainMeshProvider.variables(), ", ") << endl;

    // Open CGNS output file; create base node
    auto out_file = cgcall(
        cg_open,
        [&]{ return "Failed to open output file '" + outputName + "'"; },
        outputName.c_str(), CG_MODE_WRITE);
    auto coordVar = mainMeshProvider.coordinateVariables();
    auto dim = coordVar.size();
    auto out_base = CGCALL(cg_base_write, out_file, "base", dim, dim);

    // Write mesh (nodes and elements)
    REPORT_PROGRESS_STAGE("Write Mesh");
    auto cache = mainMeshProvider.makeCache();
    std::vector<int> zoneIds;
    for (auto zone : mainMeshProvider.zones(cache, coordVar))
    {
        auto zoneName = string("zone_") + to_string(zoneIds.size());
        REPORT_PROGRESS_STAGE(zoneName);
        auto nodeCount = zone.nodeCount();
        BOOST_ASSERT(nodeCount > 0);
        auto elementCount = zone.elementCount();
        cgsize_t size[] = {
            static_cast<cgsize_t>(nodeCount),
            static_cast<cgsize_t>(elementCount),
            0
        };
        auto out_zone = CGCALL(cg_zone_write, out_file, out_base, zoneName.c_str(), size, Unstructured);
        zoneIds.push_back(out_zone);
        writeZoneNodes(out_file, out_base, out_zone, zone, dim);
        writeZoneElements(out_file, out_base, out_zone, zone);
    }

    REPORT_PROGRESS_STAGE("Write fields");
    auto fieldVar = mainMeshProvider.fieldVariables();
    auto izone = 0;
    auto fieldNames = mainMeshProvider.variables();
    for (auto zone : mainMeshProvider.zones(cache, fieldVar))
    {
        auto zoneName = string("zone_") + to_string(zoneIds.size());
        REPORT_PROGRESS_STAGE(zoneName);
        writeZoneFields(out_file, out_base, zoneIds.at(izone), zone, fieldVar, fieldNames);
        ++izone;
    }

    if (cg_close(out_file) != CG_OK)
        throw runtime_error("Failed to close file '" + mainMeshFileName + "'");
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
