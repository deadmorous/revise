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

    string processingModeString = "mesh";
    po::options_description po_generic("Gerneric options");
    po_generic.add_options()
            ("help,h", "Produce help message")
            ("quiet,q", "Don't display progress of separate operations")
            ("threads,t", po_value(result.threadCount), "Use the specified number of threads of execution");
    po::options_description po_jobctl("Job control options");
    po_jobctl.add_options()
            ("mode", po_value(processingModeString), enumHelp<ProcessingMode>("Processing mode"));
    po::options_description po_mesh_proc("Mesh processing options");
    po_mesh_proc.add_options()
            ("output_dir", po_value(result.outputDirectory), "Output directory")
            ("mesh_file", po::value(&result.meshFileName), "Mesh file name")
            ("depth", po_value(result.metadataBlockDepth), "Maximal depth within one block")
            ("max_full_level", po_value(result.metadataMaxFullLevel), "Maximal full metadata level")
            ("max_level", po_value(result.metadataMaxLevel), "Maximal level of a metadata block")
            ("refine", po_value(result.refinerParam), "Anisotropic element refinement (0=max, 1=none)")
            ("brefine", po_value(result.boundaryRefine), "Additional refinement at boundary (positive integer)")
            ("dim", po_value(result.spaceDimension), "Space dimension (1, 2, or 3)")
            ("boundary", po_value(result.extractBoundary), "Extract mesh boundary")
            ("btec", po_value(result.saveBoundaryTecplot), "Save boundary to a tecplot file")
            ("exact_problem_id", po_value(result.exactProblemId), "Identifier of exact solution to generate")
            ("exact_config", po_value(result.exactConfigFileName), "Configuration file describing an exact solution to generate")
            ("exact_cells", po_value(result.exactCellCount), "Domain cell count for the exact solution to generate")
            ("exact_time_steps", po_value(result.exactTimeStepCount), "Time step count for the exact solution to generate")
            ("print_exact_config", "Print exact solution config (example if no file specified)");

    po::positional_options_description po_pos;
    po_pos.add("mesh_file", 1);

    po::variables_map vm;
    auto po_alloptions = po::options_description().add(po_generic).add(po_jobctl).add(po_mesh_proc);
    po::store(po::command_line_parser(argc, argv)
              .options(po_alloptions)
              .positional(po_pos).run(), vm);
    po::notify(vm);

    result.processingMode = silver_bullets::enum_item_value<ProcessingMode>(processingModeString);

    if (vm.count("help")) {
        cout << "Usage: s3dmm_prep [options ...]" << endl;
        cout << po_alloptions << "\n";
        result.printHelpOnly = true;
    }

    if (vm.count("print_exact_config"))
        result.printExactConfig = true;

    result.quiet = vm.count("quiet") > 0;

    if (!result.printHelpOnly) {
        switch (result.processingMode) {
        case MeshProcessing:
        case DomainMerging:
        case MeshInfo:
            if (result.meshFileName.empty())
                throw runtime_error("No mesh file name is specified");
            break;
        case ExactSolutionProcessing:
            break;
        }
    }
    return result;
}
