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

#include "BlockTreeNodes.hpp"
#include "BlockTree_io.hpp"
#include "MultiIndex_binary_io.hpp"

namespace s3dmm {

template <unsigned int N, class BT, class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>& operator<<(
        BinaryWriterTemplate<S, VS, SS>& writer, const BlockTreeNodes<N, BT>& btn)
{
    auto& data = btn.data();
    writer << data.root << data.maxDepth << data.n2i;
    return writer;
}

template <unsigned int N, class BT, class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>& operator>>(
        BinaryReaderTemplate<S, VS, SS>& reader, BlockTreeNodes<N, BT>& btn)
{
    typename BlockTreeNodes<N, BT>::Data data;
    reader >> data.root >> data.maxDepth >> data.n2i;
    btn = std::move(BlockTreeNodes<N, BT>::fromData(std::move(data)));
    return reader;
}

} // s3dmm
