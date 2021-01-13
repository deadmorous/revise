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

#include <ostream>

#include <boost/assert.hpp>

class PrefixWriter
{
public:
    PrefixWriter(std::ostream& s) : m_s(s)
    {}

    void allocPrefix()
    {
        m_pos = m_s.tellp();
        std::size_t placeholder = 0;
        m_s.write(reinterpret_cast<const char*>(&placeholder), sizeof(std::size_t));
    }

    void writeSizePrefix() {
        writePrefix([this](auto pos) {
            return static_cast<std::size_t>(pos) -
                   (static_cast<std::size_t>(m_pos) + sizeof(std::size_t));
        });
    }

    void writePosPrefix() {
        writePrefix([](auto pos) { return pos; });
    }

private:
    std::ostream& m_s;
    std::streamsize m_pos = ~0;

    template<class F>
    void writePrefix(const F& f)
    {
        BOOST_ASSERT(m_pos != ~0);
        auto pos = m_s.tellp();
        m_s.seekp(m_pos);
        auto prefix = static_cast<std::size_t>(f(pos));
        m_s.write(reinterpret_cast<const char*>(&prefix), sizeof(std::size_t));
        m_s.seekp(pos);
        m_pos = ~0;
    }
};
