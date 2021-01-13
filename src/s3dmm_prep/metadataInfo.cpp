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

#include "s3dmm/Metadata.hpp"
#include "s3dmm/IndexTransform.hpp"

#include "filename_util.hpp"
#include "TsvWriter.hpp"

#include "silver_bullets/templatize/resolve_template_args.hpp"

#include <iostream>
#include <fstream>

using namespace std;
using namespace s3dmm;

namespace {

template <unsigned int N>
void metadataInfoTemplate(const RunParameters& param)
{
    using BT = typename Metadata<N>::BT;
    using BTN = BlockTreeNodes<N, BT>;

    auto mainMeshFileName = firstOutputFrameFileName(param.meshFileName).first;
    auto metadataFileName = mainMeshFileName + ".s3dmm-meta";
    ifstream s(metadataFileName, ios::binary);
    Metadata<N> md(s);
    auto& bt = md.blockTree();
    TsvWriter w(cout);
    auto printNheaders = [&w](const string& prefix, char label) {
        for (auto d=0u; d<N; ++d)
            w << prefix + char(label+d);
    };
    w << "level"
      << "index";
    printNheaders("location_", 'x');
    w << "depth"
      << "nodes"
      << "fill"
      << endl;
    for (auto& level : md.levels()) {
        for (auto& mdBlock : level) {
            auto blockId = bt.blockAt(mdBlock.blockIndex, mdBlock.level);
            auto depth = md.subtreeDepth(blockId);
            auto btn = md.blockTreeNodes(blockId);

            auto actualBlockNodes = btn.data().n2i.size();
            auto denseBlockNodes = IndexTransform<N>::vertexCount(depth);
            auto blockFillRatio = static_cast<real_type>(actualBlockNodes) / denseBlockNodes;

            w << blockId.level
              << blockId.index
              << blockId.location
              << depth
              << actualBlockNodes
              << blockFillRatio
              << endl;
        }
    }
}

struct callMetadataInfoTemplate {
    template<unsigned int N> void operator()(const RunParameters& param) const {
        metadataInfoTemplate<N>(param);
    }
};

} // anonymous namespace

void metadataInfo(const RunParameters& param)
{
    REPORT_PROGRESS_STAGES();
    REPORT_PROGRESS_STAGE("Run the entire job");
    silver_bullets::resolve_template_args<
        integer_sequence<unsigned int, 1,2,3>>(
        make_tuple(param.spaceDimension), callMetadataInfoTemplate(), param);
}
