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

#include "meshInfo.hpp"
#include "RunParameters.hpp"

#include "s3dmm/MeshDataProvider.hpp"

#include "foreach_byindex32.hpp"
#include "filename_util.hpp"

#include <iostream>

#include <boost/algorithm/string/join.hpp>

using namespace std;
using namespace s3dmm;

SILVER_BULLETS_BEGIN_DEFINE_ENUM_NAMES(MeshElementType)
    {MeshElementType::Triangle, "triangle"},
    {MeshElementType::Quad, "quad"},
    {MeshElementType::Tetrahedron, "tetrahedron"},
    {MeshElementType::Hexahedron, "hexahedron"}
SILVER_BULLETS_END_DEFINE_ENUM_NAMES()

void meshInfo(const RunParameters& param)
{
    string mainMeshFileName;
    bool hasTimeSteps;
    tie(mainMeshFileName, hasTimeSteps) = firstOutputFrameFileName(param.meshFileName);

    MeshDataProvider meshProvider(mainMeshFileName);

    cout << "Problem: " << param.meshFileName << endl
         << "Main mesh file: " << mainMeshFileName << endl;
    if (hasTimeSteps) {
        unsigned int frame = 0;
        for (; ; ++frame) {
            auto meshFileName = frameOutputFileName(param.meshFileName, frame, true);
            if (!experimental::filesystem::exists(meshFileName))
                break;
        }
        cout << "Time steps: " << frame << endl;
    }
    else
        cout << "No time steps" << endl;
    cout << endl;

    cout << "Variables:" << endl
         << boost::join(meshProvider.variables(), "\t") << endl << endl;

    struct ZoneInfo {
        size_t elementCount;
        size_t nodeCount;
        MeshElementType elementType;
    };
    vector<ZoneInfo> zoneInfo;

    auto cache = meshProvider.makeCache();
    for (auto zone : meshProvider.zones(cache)) {
        zoneInfo.push_back({
            zone.elementCount(),
            zone.nodeCount(),
            zone.elementType()
        });
    }

    cout << "zone\telement_type\telements\tnodes" << endl;
    size_t totalElementCount = 0;
    size_t totalNodeCount = 0;
    foreach_byindex32(izone, zoneInfo) {
        auto& zi = zoneInfo[izone];
        cout << izone << '\t'
             << silver_bullets::enum_item_name(zi.elementType) << '\t'
             << zi.elementCount << '\t'
             << zi.nodeCount << endl;
        totalElementCount += zi.elementCount;
        totalNodeCount += zi.nodeCount;
    }
    cout << "---- Total ----" << endl
         << "zones: " << zoneInfo.size() << ", "
         << "elements: " << totalElementCount << ", "
         << "nodes: " << totalNodeCount << endl;
}
