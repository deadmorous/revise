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
#include <experimental/filesystem>

namespace s3dmm {

inline std::tuple<std::string, std::string, std::string> splitFileName(const std::string& baseName)
{
    using namespace std::experimental::filesystem;
    auto p = path(baseName);
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
    return experimental::filesystem::path(dir).append(stem).append(oss.str());
}

inline bool baseNameCorrespondsToFile(const std::string& baseName)
{
    using namespace std::experimental::filesystem;
    return exists(baseName) && is_regular_file(baseName);
}

inline bool baseNameCorrespondsToDirectory(const std::string& baseName)
{
    using namespace std::experimental::filesystem;
    auto s = splitFileName(baseName);
    std::string dirName = path(std::get<0>(s)).append(std::get<1>(s));
    return exists(dirName) && is_directory(dirName);
}

inline std::string outputFrameDirectory(const std::string& baseName, bool hasTimeFrames)
{
    using namespace std;
    using namespace experimental::filesystem;
    auto s = splitFileName(baseName);
    if (hasTimeFrames)
        return path(get<0>(s)).append(get<1>(s));
    else
        return get<0>(s);
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

} // s3dmm

