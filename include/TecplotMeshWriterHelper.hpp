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

#include <iostream>
#include <vector>
#include "real_type.hpp"

struct TecplotMeshWriterHelper
{
public:
    explicit TecplotMeshWriterHelper(std::ostream& s) : m_s(s) {}
    void writeFileHeader (const std::vector<std::string>& fields)
    {
        m_s << "variables = ";
        for (auto& field : fields)
            m_s << ' ' << field;
        m_s << std::endl;
    }
    void writeZoneHeader (unsigned int i)
    {
        m_s << "zone  i = " << i << std::endl
            << "T=\"  block1       \"" << std::endl;
    }
    void writeZoneHeader (unsigned int i, unsigned int j)
    {
        m_s << "zone  i = " << i << " j = " << j << std::endl
            << "T=\"  block1       \"" << std::endl;
    }
    void writeZoneHeader (unsigned int i, unsigned int j, unsigned int k)
    {
        m_s << "zone  i = " << i << " j = " << j << " k = " << k << std::endl
            << "T=\"  block1       \"" << std::endl;
    }

    void writeRow(s3dmm::real_type firstArg) {
        m_s << firstArg;
        writeEmptyRow();
    }
    template<class ... RestArgs>
    void writeRow(s3dmm::real_type firstArg, s3dmm::real_type secondArg, RestArgs ... restArgs) {
        m_s << firstArg << '\t';
        writeRow(secondArg, restArgs ...);
    }
private:
    void writeEmptyRow() {
        m_s << std::endl;
    }
    std::ostream& m_s;
};

