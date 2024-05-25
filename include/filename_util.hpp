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

#include <string>
#include <utility>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace s3dmm {

inline std::tuple<std::string, std::string, std::string> splitFileName(const std::string& baseName)
{
    auto p = std::filesystem::path(baseName);
    return { p.parent_path(), p.stem(), p.extension() };
}

inline std::string frameOutputFileName(const std::string& baseName, unsigned int frameNumber, bool hasTimeSteps)
{
    using namespace std;
    if (!hasTimeSteps)
        return  baseName;
    string dir, stem, ext;
    tie(dir, stem, ext) = splitFileName(baseName);
    ostringstream oss;
    oss << stem << "_t"
        << setfill('0') << setw(6) << frameNumber
        << ext;
    return filesystem::path(dir).append(stem).append(oss.str());
}

inline bool baseNameCorrespondsToFile(const std::string& baseName)
{
    namespace fs = std::filesystem;
    return fs::exists(baseName) && fs::is_regular_file(baseName);
}

inline bool baseNameCorrespondsToDirectory(const std::string& baseName)
{
    namespace fs = std::filesystem;
    auto s = splitFileName(baseName);
    std::string dirName = fs::path(std::get<0>(s)).append(std::get<1>(s));
    return fs::exists(dirName) && fs::is_directory(dirName);
}

inline std::string outputFrameDirectory(const std::string& baseName, bool hasTimeFrames)
{
    namespace fs = std::filesystem;
    auto s = splitFileName(baseName);
    if (hasTimeFrames)
        return fs::path(std::get<0>(s)).append(std::get<1>(s));
    else
        return std::get<0>(s);
}

inline std::string outputFrameDirectory(const std::string& baseName)
{
    using namespace std;
    if (baseNameCorrespondsToFile(baseName))
        return outputFrameDirectory(baseName, false);
    else if (baseNameCorrespondsToDirectory(baseName))
        return outputFrameDirectory(baseName, true);
    else
        throw invalid_argument(string("Failed to find input file(s) matching the specified name '") + baseName + "'");
}

inline std::pair<std::string, bool> firstOutputFrameFileName(const std::string& baseName)
{
    using namespace std;
    if (baseNameCorrespondsToFile(baseName))
        return { baseName, false };
    else if (baseNameCorrespondsToDirectory(baseName))
        return { frameOutputFileName(baseName, 0, true), true };
    else
        throw invalid_argument(string("Failed to find input file(s) matching the specified name '") + baseName + "'");
}

inline std::string s3dmmBaseName(const std::string& inputBaseName, const std::string& outputDirectory)
{
    namespace fs = std::filesystem;

    if (outputDirectory.empty())
        return inputBaseName;

    if (std::error_code ec; !fs::exists(outputDirectory) &&
                            !fs::create_directories(outputDirectory, ec))
    {
        std::ostringstream oss;
        oss << "Failed to create otuput directory '" << outputDirectory << "': " << ec.message();
        throw std::runtime_error(oss.str());
    }

    auto hasTimeSteps =
        baseNameCorrespondsToDirectory(inputBaseName);

    if (!hasTimeSteps && !baseNameCorrespondsToFile(inputBaseName))
        throw std::invalid_argument(
            "Failed to find input file(s) matching the specified name '" + inputBaseName + "'");

    auto [dir, stem, ext] = splitFileName(inputBaseName);

    if (hasTimeSteps)
        return outputDirectory + ext;
    else
        return fs::path(outputDirectory) / (stem + ext);
}

} // s3dmm

