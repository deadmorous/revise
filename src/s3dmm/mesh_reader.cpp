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

#include "s3dmm/mesh_reader.hpp"
#include "BinaryMeshReader.hpp"
#include "TsagiMeshReader.h"
#include "Imamod_gMeshReader.h"

namespace s3dmm {

tecplot_rw::ITecplotReaderPtr CreateBinaryMeshReader()
{
    return std::make_shared<BinaryMeshReader>();
}

tecplot_rw::ITecplotReaderPtr CreateTsagiMeshReader()
{
    return std::make_shared<TsagiMeshReader>();
}

tecplot_rw::ITecplotReaderPtr CreateImamod_gMeshReader()
{
    return std::make_shared<Imamod_gMeshReader>();
}

} // s3dmm
