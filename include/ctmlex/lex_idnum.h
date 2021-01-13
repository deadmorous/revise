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

// lex_idnum.h

/** \ingroup CTMSTD_LANG*/
/** \defgroup CTMSTD_LANG_LEX General lexical analysis routines
This module implements several commonly used functions for lexical analysis.

<b> Supported platforms:</b> Linux, Win32.

<b> Module type:</b> dynamic library.

<b> Linkage instructions</b>

Use early binding to link the module to your C++ program.
    - In Linux builds, pass the \b ctmlex.so shared library to the linker along with your object files.
    - In Win32 builds, pass the \b ctmlex.lib export library to the linker along with your object files.
*/

/// \ingroup CTMSTD_LANG_LEX
/// \file lex_idnum.h
/// \brief \ref CTMSTD_LANG_LEX

#pragma once

namespace ctm {

/** \ingroup CTMSTD_LANG*/
/// \brief \ref CTMSTD_LANG
namespace lang {

#ifdef _WIN32
#ifdef CTM_LANG_CTMLEX_EXPORTS
#define CTM_LANG_CTMLEX_API __declspec(dllexport)
#else
#define CTM_LANG_CTMLEX_API __declspec(dllimport)
#endif
#else // _WIN32
#define CTM_LANG_CTMLEX_API
#endif // _WIN32

/** \addtogroup CTMSTD_LANG_LEX*/
//@{

/// Returns true if \a str is a C/C++ identifier.
bool CTM_LANG_CTMLEX_API IsIdentifier( const char *str );

/// Returns true if \a str is a real number.
bool CTM_LANG_CTMLEX_API IsNumber( const char *str );

/// Returns true if \a str is an integer number.
bool CTM_LANG_CTMLEX_API IsIntegerNumber( const char *str );

/// Returns true if \a str is a hexadecimal integer number.
bool CTM_LANG_CTMLEX_API IsHexIntegerNumber( const char *str );

//@}

} // end namespace lang
} // end namespace ctm

