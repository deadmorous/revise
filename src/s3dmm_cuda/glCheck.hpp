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

#include <GL/gl.h>
#include <sstream>

#define GL_CHECK() glCheck(__FILE__, __LINE__)

inline void glCheck(const char *file, int line) {
    auto e = glGetError();
    if (e != GL_NO_ERROR) {
        std::ostringstream oss;
        oss << "GL error " << e << " in " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}
