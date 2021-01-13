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

#include "hires_time.h"

namespace s3dmm
{

template<class TimeType>
struct FieldTimestampsTemplate
{
    TimeType afterReadSparseField = 0;
    TimeType afterComputeDenseField = 0;
    TimeType afterGetFieldRange = 0;

    bool empty() const
    {
        return
            afterReadSparseField == 0 &&
            afterComputeDenseField == 0 &&
            afterGetFieldRange == 0;
    }
};

using FieldTimestamps = FieldTimestampsTemplate<s3dmm::hires_time_t>;

} // namespace s3dmm
