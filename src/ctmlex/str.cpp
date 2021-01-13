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

#include "ctmlex/lex_str.h"

using namespace std;

namespace ctm {
namespace lang {

string CTM_LANG_CTMLEX_API AddEscapeSeq( const string& str_, bool bEscapeSingleQuote )
{
	if ( str_.empty() ) return str_;
	string str;
	for (const char* p=str_.c_str(); (p && *p); p++)
	{
		switch (*p)
		{
		case '\n':
			str+="\\n";
			break;
		case '\r':
			str+="\\r";
			break;
		case '\t':
			str+="\\t";
			break;
		case '\'':
            if( !bEscapeSingleQuote ) {
                str += *p;
                break;
                }
		case '"':
		case '\\':
			str += '\\';
			str += *p;
			break;
		default:
			str += *p;
		};
	};
	//str+='"';
	return str;
};

string CTM_LANG_CTMLEX_API RemoveEscapeSeq( const string& str_ )
{
	if ( str_.empty() ) return str_;
	string tmp;
	unsigned int n=str_.size();
	for (const char* p=&str_[0]; (p && *p); p++)
	{
		if (*p!='\\') { tmp+=*p; continue; };
		p++;
		switch (*p)
		{
		case 'n':
			tmp+='\n';
			break;
		case 'r':
			tmp+='\r';
			break;
		case 't':
			tmp+='\t';
			break;
		case '\\':
		case '"':
		case '\'':
			tmp+=*p;
			break;
		case 0:
			// The only one backslash at the end of string
			// Just ommit it and leave.
			// Dont know, whether it is right.
			break;
		default:
			tmp+=*p; // Microsoft specific behaviour
		};
	};
	return tmp;
};

} // lang
} // ctm



