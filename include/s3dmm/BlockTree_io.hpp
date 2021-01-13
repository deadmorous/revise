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

#include "BlockTree.hpp"
#include "CompressedBlockTree.hpp"
#include "BoundingCube_io.hpp"
#include "MultiIndex_binary_io.hpp"

namespace s3dmm {

template <unsigned int N, class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const detail::TreeBlockId<N>& blockId)
{
    writer << blockId.index << blockId.level << blockId.location;
    return writer;
}

template <unsigned int N, class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, detail::TreeBlockId<N>& blockId)
{
    reader >> blockId.index >> blockId.level >> blockId.location;
    return reader;
}

template <unsigned int N, class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const BlockTree<N>& bt)
{
    auto& data = bt.data();
    writer << data.bc << data.children;
    return writer;
}

template <unsigned int N, class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, BlockTree<N>& bt)
{
    typename BlockTree<N>::Data data;
    reader >> data.bc >> data.children;
    bt = std::move(BlockTree<N>::fromData(std::move(data)));
    return reader;
}

template <unsigned int N, class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const CompressedBlockTree<N>& cbt)
{
    auto& data = cbt.data();
    writer << data.bc << data.dataBitCount << data.data;
    return writer;
}

template <unsigned int N, class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, CompressedBlockTree<N>& cbt)
{
    typename CompressedBlockTree<N>::Data data;
    reader >> data.bc >> data.dataBitCount >> data.data;
    cbt = std::move(CompressedBlockTree<N>::fromData(std::move(data)));
    return reader;
}

} // s3dmm
