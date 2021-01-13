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

#include "VsWorkerInit.hpp"

#include "silver_bullets/fs_ns_workaround.hpp"

#include <vlCore/FileSystem.hpp>
#include <vlCore/VisualizationLibrary.hpp>

namespace s3vs
{

void VsWorkerInit::initialize(
    const std::string& shaderPath,
    bool logInfo,
    const std::string& logFileName)
{
    namespace fs = std::filesystem;

    std::lock_guard<mutex_type> lock(VsWorkerInit::getMutex());

    if (vl::VisualizationLibrary::isCoreInitialized())
        return;

    if (!logFileName.empty()) {
        std::string dir = fs::path(logFileName).parent_path();
        if (!fs::exists(dir) && !fs::create_directories(dir))
            throw std::runtime_error(std::string("Failed to create path '" + dir + "' for vl log files"));
    }
    setenv("VL_LOGFILE_PATH", logFileName.c_str(), true);
    vl::VisualizationLibrary::init(logInfo);
    vl::defFileSystem()->directories().push_back(
        new vl::DiskDirectory(shaderPath));

    vl::Log::setLogMutex(&getIMutex());
}

} // namespace s3vs
