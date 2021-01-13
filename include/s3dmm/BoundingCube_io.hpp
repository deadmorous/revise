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

#include "BoundingCube.hpp"
#include "binary_io.hpp"

namespace s3dmm {

template <unsigned int N, class T, class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const BoundingCube<N, T>& bc)
{
    writer << bc.min() << bc.size();
    return writer;
}

template <unsigned int N, class T, class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, BoundingCube<N, T>& bc)
{
    typename BoundingCube<N, T>::vector_type bcMin;
    T bcSize;
    reader >> bcMin >> bcSize;
    bc = BoundingCube<N, T>(bcMin, bcSize);
    return reader;
}

} // s3dmm

