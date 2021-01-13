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

#include "MultiIndex.hpp"

namespace s3dmm {

class TsvWriter
{
public:
    explicit TsvWriter(std::ostream& s) : m_s(s) {}

    template<class T>
    void write(const T& x) {
        if (m_startOfLine)
            m_startOfLine = false;
        else
            m_s << '\t';
        m_s << x;
    }

    void newline() {
        m_s << std::endl;
        m_startOfLine = true;
    }

    std::ostream& stream() const {
        return m_s;
    }

private:
    std::ostream& m_s;
    bool m_startOfLine = true;
};

template<class T>
inline TsvWriter& operator<<(TsvWriter& w, const T& x)
{
    w.write(x);
    return w;
}

template<class T, unsigned int N>
inline TsvWriter& operator<<(TsvWriter& w, const MultiIndex<N, T>& x)
{
    for (auto i=0u; i<N; ++i)
        w << x[i];
    return w;
}

inline TsvWriter& operator<<(TsvWriter& w, std::ostream& (*f)(std::ostream&))
{
    if (f == static_cast<std::ostream& (*)(std::ostream&)>(std::endl))
        w.newline();
    else
        f(w.stream());
    return w;
}

} // s3dmm
