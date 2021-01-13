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

#include "lex_idnum.h"
#include <string>

namespace ctm {
namespace lang {

std::string CTM_LANG_CTMLEX_API AddEscapeSeq( const std::string& str, bool bEscapeSingleQuote = true );
std::string CTM_LANG_CTMLEX_API RemoveEscapeSeq( const std::string& str );


inline std::string DecorateString( const std::string& str )
{
	return "\"" + AddEscapeSeq( str, false ) + "\"";
};

inline std::string UndecorateString( const std::string& str )
{
	std::string s = str;
	if (!s.empty() && (s[0]=='"')) s.erase(0,1);
	if (!s.empty() && (s[s.size()-1]=='"')) s.erase(s.size()-1,1);
	return RemoveEscapeSeq( s );
};

inline size_t FindNotEscSymbol( const std::string& s, char ch, size_t spos = 0 )
{
	size_t pos = spos;
	for ( ; ( ( pos = s.find_first_of( ch, pos ) ) != std::string::npos ); ++pos )
	{
		size_t n = 0;
		for ( ; ( ( ( pos - spos ) > n ) && ( s[pos - 1 - n] == '\\' ) ); ++n );
		if ( ( n%2 ) == 0 )
			break;
	};
	return pos;
};

inline size_t FindDQuot( const std::string& s, size_t spos = 0 )
{
	return FindNotEscSymbol( s, '"', spos );
};


} // lang namespace
} // ctm namespace

