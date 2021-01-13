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
#include <stdexcept>
// #include <experimental/filesystem>

#include "RunParameters.hpp"

#include "vtk_rect_grid.hpp"

using namespace std;

namespace {

void run(const RunParameters& param) {
    switch (param.contentType) {
    case VtkFileContentType::RectlilinearGrid:
        convertVtkRectGrid(param);
        break;
    }
}

} // anonymous namespace

int main(int argc, char *argv[])
{
    try {
        auto param = RunParameters::parse(argc, argv);
        if (!param.printHelpOnly)
            run(param);
        return EXIT_SUCCESS;
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
