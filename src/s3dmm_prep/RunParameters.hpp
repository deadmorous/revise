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

#include "ProcessingMode.hpp"
#include "real_type.hpp"

struct RunParameters
{
    bool printHelpOnly = false;
    bool quiet = false;
    unsigned int threadCount = 1u;

    ProcessingMode processingMode = MeshProcessing;

    unsigned int spaceDimension = 3u;
    unsigned int metadataBlockDepth = 7;
    unsigned int metadataMaxFullLevel = 0;
    unsigned int metadataMaxLevel = 20;

    // Mesh processing specific parameters
    std::string meshFileName;
    s3dmm::real_type refinerParam = s3dmm::make_real(1);
    bool extractBoundary = true;
    unsigned int boundaryRefine = 1u;
    bool saveBoundaryTecplot = false;

    // Exact solution specific parameters
    std::string exactProblemId;
    std::string exactConfigFileName;
    std::string exactOutputDirectory;
    std::string exactCellCount;
    std::string exactTimeStepCount;
    bool printExactConfig = false;

    static RunParameters parse(int argc, char *argv[]);
};
