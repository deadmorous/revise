#include <QApplication>
#include <QMessageBox>

#include <iostream>
#include <regex>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#include "MainWindow.h"

using namespace std;

template<class T, class Compound>
inline boost::program_options::typed_value<T>* po_value(Compound& sd) {
    return boost::program_options::value(&static_cast<T&>(sd));
}

bool setProgramOptions(StateData& stateData, int argc, char *argv[])
{
    namespace po = boost::program_options;
    po::options_description po_generic("Gerneric options");
    po_generic.add_options()
            ("help,h", "produce help message");
    po::options_description po_basic("Main options");
    string subtreeRootIdString;
    po_basic.add_options()
            ("mesh_file", po_value<WithInputFileName>(stateData), "Mesh file name")
            ("field", po_value<WithFieldName>(stateData), "Field name")
            ("qt", po_value<WithDisplayQtree>(stateData), "Display quad tree")
            ("qtcompact", po_value<WithQtreeCompactView>(stateData), "Compact view of quad tree")
            ("qtids", po_value<WithQtreeDisplayIds>(stateData), "Display ids of quad tree nodes")
            ("qtlim", po_value<WithLimitedQtree>(stateData), "Limit quadtree to the root subtree")
            ("mesh", po_value<WithDisplayMesh>(stateData), "Display original mesh")
            ("boundary", po_value<WithDisplayBoundary>(stateData), "Display original mesh boundary")
            ("bnids", po_value<WithDisplayBoundaryNodeNumbers>(stateData), "Display boundary node numbers")
            ("bdirmk", po_value<WithDisplayBoundaryDirMarkers>(stateData), "Display boundary direction markers")
            ("sn", po_value<WithDisplaySparseNodes>(stateData), "Display sparse nodes")
            ("snids", po_value<WithDisplaySparseNodeNumbers>(stateData), "Display numbers of sparse nodes")
            ("df", po_value<WithDisplayDenseField>(stateData), "Display dense field")
            ("fill", po_value<WithFill>(stateData), "Draw filled areas")
            ("qfill", po_value<WithQuadtreeNodeFill>(stateData), "Draw filled quad tree nodes")
            ("root", po::value(&subtreeRootIdString), "Root subtree id (i,j,level) to start from");

    po::positional_options_description po_pos;
    po_pos.add("mesh_file", 1);

    po::variables_map vm;
    auto po_alloptions = po::options_description().add(po_generic).add(po_basic);
    po::store(po::command_line_parser(argc, argv)
              .options(po_alloptions)
              .positional(po_pos).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << po_alloptions << "\n";
        return false;
    }

    if (!subtreeRootIdString.empty()) {
        // Parse output subtree id
        regex rxSubtree("^(\\d+),(\\d+),(\\d+)$");
        smatch m;
        if (!regex_match(subtreeRootIdString, m, rxSubtree))
            throw runtime_error("Invalid subtree id");
        BOOST_ASSERT(m.size() == 4);
        unsigned int n[3];
        transform(m.begin()+1, m.end(), n, [](const auto& x) {
            return boost::lexical_cast<unsigned int>(x);
        });
        stateData.setBlockTreeLocation({n[2], {n[0], n[1]}});
    }

    return true;
}

int main(int argc, char *argv[])
{
    using namespace std;

    QApplication app(argc, argv);
    try {
        MainWindow mainWindow;

        if (!setProgramOptions(mainWindow.stateData(), argc, argv))
            return EXIT_SUCCESS;

        mainWindow.show();
        return app.exec();
    }
    catch(exception& e) {
        QMessageBox::critical(
                    nullptr,
                    QApplication::applicationName() + " - critical error",
                    QString::fromUtf8(e.what()));
        cerr << e.what() << endl;
        return EXIT_FAILURE;
    }
}
