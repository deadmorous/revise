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

// ctmlex.cpp : Defines the entry point for the DLL application.
//

#include "ctmlex/lex_idnum.h"
#include <ctype.h>
#include <string.h>

namespace ctm {
namespace lang {

bool CTM_LANG_CTMLEX_API IsIdentifier( const char *str )
    {
    if( !str )   return false;
    if( !( isalpha( *str )   ||   *str == '_' ) )   return false;
    for( int i=1; str[i]; i++ )
        if( !( isalnum( str[i] )   ||   str[i] == '_' ) )   return false;
    return true;
    }

bool CTM_LANG_CTMLEX_API IsNumber( const char *str )
    {
    str += strspn( str, " \t" );
    int nMantissaDigits = 0;
    if( *str && strchr( "+-", *str ) )   str++;
    while( isdigit( *str ) )   { nMantissaDigits++; str++; }
    if( *str == '.' )
        { str++; while( isdigit( *str ) )   { nMantissaDigits++; str++; } }
    if( nMantissaDigits == 0 )   return false;
    if( *str == 0 )   return true;
    if( toupper(*str) != 'E' )   return false;
    str++;
	// >> AK 070505 handle expresions such 1e
	if ( !*str ) return false; 
	// << AK
    if( strchr( "+-", *str ) )   str++;
    int nExpDigits = 0;
    while( isdigit( *str ) )   { nExpDigits++; str++; }
    return nExpDigits > 0   &&   nExpDigits < 4   &&   *str == 0;
    }

bool CTM_LANG_CTMLEX_API IsIntegerNumber( const char *str )
    {
    str += strspn( str, " \t" );
    if( *str && strchr( "+-", *str ) )   str++;
    int n = 0;
    while( isdigit( *str ) )   { n++; str++; }
    return n > 0   &&   *str == 0;
    }

bool CTM_LANG_CTMLEX_API IsHexIntegerNumber( const char *str )
    {
    str += strspn( str, " \t" );
    int n = 0;
    while( isxdigit( *str ) )   { n++; str++; }
    return n > 0   &&   *str == 0;
    }

} // end namespace lang
} // end namespace ctm



