/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021-2024 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

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

#include "OpenRange.hpp"

namespace s3dmm
{

struct IndexRangeSplit
{
    IndexRange before;
    unsigned int at;
    IndexRange after;

    bool operator==(const IndexRangeSplit& that) const noexcept = default;
};

inline std::ostream& operator<<(std::ostream& s, const IndexRangeSplit& rs)
{
    s << "IndexRangeSplit{ " << rs.before << " / ";
    if (rs.at == ~0u)
        s << "empty";
    else
        s << rs.at;
    return s << " / " << rs.after << " }";
}


} // namespace s3dmm
