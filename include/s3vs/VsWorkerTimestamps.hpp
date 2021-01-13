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

#include "s3dmm/FieldTimestamps.hpp"
#include "TsvWriter.hpp"

#include <array>
#include <vector>

namespace s3vs
{

template<class TimeType>
struct BlockTimestampsTemplate
{
    std::array<unsigned int, 3> blockIndex;
    s3dmm::FieldTimestampsTemplate<TimeType> afterPrimaryField;
    s3dmm::FieldTimestampsTemplate<TimeType> afterSecondaryField;
    TimeType beforeRender = 0;
    TimeType afterRender = 0;
};

using BlockTimestamps = BlockTimestampsTemplate<s3dmm::hires_time_t>;

template<class TimeType>
struct VsWorkerTimestampsTemplate
{
    unsigned int level = 0;
    TimeType start = 0;
    TimeType afterClearViewport = 0;
    TimeType afterSortBlocks = 0;
    std::vector<BlockTimestampsTemplate<TimeType>> blocks;
    TimeType afterRenderDenseField = 0;
    TimeType afterGetClipRect = 0;
    TimeType afterClip = 0;
};

using VsWorkerTimestamps = VsWorkerTimestampsTemplate<s3dmm::hires_time_t>;

template<class TimeType>
inline std::ostream& operator<<(std::ostream& s, const VsWorkerTimestampsTemplate<TimeType>& t)
{
    s3dmm::TsvWriter w(s);
    w << t.level
      << t.start
      << t.afterClearViewport
      << t.afterSortBlocks
      << t.blocks.size();

    auto writeField = [&w](char fieldType, const s3dmm::FieldTimestamps& fts) {
        w << fieldType
          << fts.afterReadSparseField
          << fts.afterComputeDenseField
          << fts.afterGetFieldRange;
    };

    auto writeBlock = [&](const BlockTimestamps& b) {
        w << 'b'
          << b.blockIndex[0] << b.blockIndex[1] << b.blockIndex[2];
        writeField('p', b.afterPrimaryField);
        writeField('s', b.afterSecondaryField);
        w << b.beforeRender << b.afterRender;
    };

    for (auto& b : t.blocks)
        writeBlock(b);
    w << 'f'
      << t.afterRenderDenseField
      << t.afterGetClipRect
      << t.afterClip;
    return s;
}

template<class TimeType>
inline std::istream& operator>>(std::istream& s, VsWorkerTimestampsTemplate<TimeType>& t)
{
    t = VsWorkerTimestampsTemplate<TimeType>();
    s >> t.level
      >> t.start
      >> t.afterClearViewport
      >> t.afterSortBlocks;

    std::size_t blockCount;
    s >> blockCount;
    t.blocks.resize(blockCount);

    auto readField = [&](s3dmm::FieldTimestampsTemplate<TimeType>& fts) {
        char fieldType;
        s >> fieldType
          >> fts.afterReadSparseField
          >> fts.afterComputeDenseField
          >> fts.afterGetFieldRange;
    };

    auto readBlock = [&](BlockTimestampsTemplate<TimeType>& b) {
        char garbage;
        s >> garbage; // 'b'
        s >> b.blockIndex[0] >> b.blockIndex[1] >> b.blockIndex[2];
        readField(b.afterPrimaryField);
        readField(b.afterSecondaryField);
        if (b.afterSecondaryField.empty())
            b.afterSecondaryField.afterReadSparseField =
                b.afterSecondaryField.afterComputeDenseField =
                b.afterSecondaryField.afterGetFieldRange =
                b.afterPrimaryField.afterGetFieldRange;

        TimeType afterReadSparseField = 0;
        TimeType afterComputeDenseField = 0;
        TimeType afterGetFieldRange = 0;

        s >> b.beforeRender >> b.afterRender;
    };

    for (auto& b : t.blocks)
        readBlock(b);
    char garbage;
    s >> garbage // 'f'
      >> t.afterRenderDenseField
      >> t.afterGetClipRect
      >> t.afterClip;
    return s;
}

} // namespace s3vs
