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

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include "s3dmm/TecplotMeshDataProvider.hpp"
#include "BinaryMeshWriterHelper.hpp"

using namespace std;

namespace {

} // anonymous namespace

int main(int argc, char *argv[])
{
    using namespace s3dmm;
    try {
        if (argc != 2)
            throw invalid_argument("Usage: tecplot2bin <tecplot-file-name.tec>");
        string tecplotFileName = argv[1];
        filesystem::path inputPath(tecplotFileName);
        if (inputPath.extension() != ".tec")
            throw invalid_argument("Input file must have the .tec filename extension");
        auto outputPath = inputPath;
        outputPath.replace_extension(".bin");
        TecplotMeshDataProvider tecplotData(tecplotFileName);
        ofstream os(outputPath, ios::binary);
        if (!os.is_open())
            throw runtime_error(string("Failed to open output file '") + string(outputPath) + "'");
        BinaryMeshWriterHelper wh(os, tecplotData.zoneCount());
        wh.writeFileHeader(tecplotData.variables());
        auto cache = tecplotData.makeCache();
        for (auto& zone : tecplotData.zones(cache)) {
            wh.writeZoneHeader(zone.structuredMeshShape());
            for (auto& node : zone.nodes())
                wh.writeRow(node.begin(), node.end());
        }
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}
