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

#include "RunParameters.hpp"

#include <boost/program_options.hpp>

#include <iostream>

#include "enumHelp.hpp"

using namespace std;

RunParameters RunParameters::parse(int argc, char *argv[])
{
    namespace po = boost::program_options;
    RunParameters result;

    auto po_value = [](auto& x) {
        return po::value(&x)->default_value(x);
    };

    string contentTypeString = silver_bullets::enum_item_name(result.contentType);
    po::options_description po_generic("Gerneric options");
    po_generic.add_options()
        ("help,h", "Produce help message")
        ("quiet,q", "Don't display progress of separate operations");
    po::options_description po_proc("Processing options");
    po_proc.add_options()
        ("content_type", po_value(contentTypeString), enumHelp<VtkFileContentType>("VTK file content type"))
        ("dim", po_value(result.spaceDimension), "Space dimension (1, 2, or 3)")
        ("input,i", po_value(result.inputFileName), "VTK input file name")
        ("output,o", po_value(result.outputFileName), "Binary output file name");

    po::variables_map vm;
    auto po_alloptions = po::options_description().add(po_generic).add(po_proc);
    po::store(po::command_line_parser(argc, argv)
                  .options(po_alloptions).run(), vm);
    po::notify(vm);

    result.contentType = silver_bullets::enum_item_value<VtkFileContentType>(contentTypeString);

    if (vm.count("help")) {
        cout << "Usage: vtk2bin [options ...]" << endl;
        cout << po_alloptions << "\n";
        result.printHelpOnly = true;
    }

    result.quiet = vm.count("quiet") > 0;

    if (!result.printHelpOnly) {
        if (result.inputFileName.empty())
            throw runtime_error("No VTK input file name is specified");
        if (result.outputFileName.empty())
            throw runtime_error("No binary output file name is specified");
    }

    return result;
}
