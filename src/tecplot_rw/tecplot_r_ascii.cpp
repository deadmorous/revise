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


#include "tecplot_rw/tecplot_reader.h"
#include <fstream>
#include <map>
#include "ctmlex/lex_str.h"
#include <stdlib.h>
#include <cassert>

using namespace std;


namespace tecplot_rw {


class XTecplotReaderASCII;
class IRecordReader;

typedef IRecordReader* (*CreateInstPtr)();

class IRecordReader
{
public:
    virtual ~IRecordReader() {}
    virtual void XProcessRecord( CreateInstPtr& pNext, string& initBufNext ) = 0;
    IRecordReader* ProcessRecord()
    {
        string initBufNext;
        CreateInstPtr pNext;
        XProcessRecord( pNext, initBufNext );
        IRecordReader* pRR = 0;
        if ( pNext )
        {
            pRR = pNext();
            pRR->Init( m_pTpR, initBufNext, *m_pStream );
        };
        return pRR;
    }
    void Init( XTecplotReaderASCII* pTpR, const string& initBuf, istream& is )
    {
        m_pTpR = pTpR;
        m_initBuf = initBuf;
        m_pStream = &is;
    }
protected:
    XTecplotReaderASCII* m_pTpR;
    string               m_initBuf;
    istream*             m_pStream;

};
//typedef cxx::MasterPtrRCX<IRecordReader> WIRR;
typedef shared_ptr<IRecordReader> WIRR;


struct ZonePosition
{
    size_t uPosVars;
    size_t uPosCnt;
    ZonePosition( size_t uPosVars_ = ~0 ) : uPosVars( uPosVars_ ), uPosCnt( ~0 ) {}
};
typedef vector< ZonePosition > ZonePosition_v;

class XTecplotReaderASCII :
    public ITecplotReader
{
public:
    virtual TecplotHeader GetHeader() const;
    virtual const ZoneInfo_v& GetZoneInfo() const;
    virtual void GetZoneVariableValues( size_t nZone, double* pbuf ) const;
    virtual void GetZoneVariableValues( size_t nZone, float* pbuf ) const;
    virtual void GetZoneConnectivity( size_t nZone, unsigned int* pbuf ) const;

    virtual void Open( const std::string& file );
    virtual void Attach( std::istream& is );
    virtual void Close();
    XTecplotReaderASCII()
    { m_pStream = 0; }

    void SetHeader( const TecplotHeader& header );
	void AddZoneInfo( const ZoneInfo& zi );
    const ZonePosition_v& GetZonePosition() const
    { return m_zonePos; }
    void AddZonePosition( const ZonePosition& zp );
    void SetZonePosition( const ZonePosition& zp, size_t nZone );
private:
    shared_ptr<istream>        m_wStream;
    istream*                   m_pStream;
    TecplotHeader              m_header;
    ZoneInfo_v                 m_zoneInfo;
    ZonePosition_v             m_zonePos;

    void ReadFileStruct();
    void Clear();

    template <typename T>
    void GetZoneVariableValuesT( size_t nZone, T* pbuf ) const;
};



class RecordReaderHeader : public IRecordReader
{
public:
    virtual void XProcessRecord( CreateInstPtr& pNext, string& initBufNext );
};


class RecordReaderZone : public IRecordReader
{
public:
    virtual void XProcessRecord( CreateInstPtr& pNext, string& initBufNext );
    static IRecordReader* CreateInst() { return new RecordReaderZone; }
};


class RecordReaderUnknown : public IRecordReader
{
public:
    virtual void XProcessRecord( CreateInstPtr& pNext, string& initBufNext );
    static IRecordReader* CreateInst() { return new RecordReaderUnknown; }
};

string ToUpper( const string& s )
{
    string s2( s.size(), ' ' );
    for ( size_t i = 0; i < s.size(); i++ )
        s2[i] = toupper( s[i] );
    return s2;
}

CreateInstPtr NewRecordStarted( const string& buf, size_t pos = 0 )
{
    typedef map< string, CreateInstPtr > S2C;
    static S2C s2c;
    if ( s2c.empty() )
    {
        const char* prec[] = { "ZONE", "TEXT", "GEOMETRY", "CUSTOMLABELS", "DATASETAUXDATA", "VARAUXDATA", 0 };
        for ( const char** p = prec; *p; p++ )
            s2c[*p] = RecordReaderUnknown::CreateInst;
        s2c["ZONE"] = RecordReaderZone::CreateInst;

    };

    size_t pos2 = buf.find_first_of( " \t,", pos );
    string s = buf.substr( pos, ( pos2 == string::npos ) ? pos2 : ( pos2 - pos ) );
    for ( size_t i = 0; i < s.length(); i++ )
        s[i] = toupper( s[i] );

    S2C::const_iterator it = s2c.find( s );
    return ( it == s2c.end() ) ? 0 : it->second;
}


struct Token
{
    string name;
    vector< string > args;
};
typedef vector< Token > v_Token;




string TokenItemNext( const string& str, size_t& pos )
{
    if ( pos == string::npos )
        return "";
    pos = str.find_first_not_of( " \t", pos );
    if ( pos == string::npos )
        return "";
    if ( str[pos] == '=' )
    {
        ++pos;
        return "=";
    };

    size_t pos2;
    if ( str[pos] == '"' )
    {
        pos2 = ctm::lang::FindDQuot( str, pos + 1 );
        if ( pos2 == string::npos )
            throw runtime_error( "Closing double quotation expected: " + str.substr( pos ) );
        ++pos2;
    }
    else
        pos2 = str.find_first_of( " \t,=", pos + 1 );
    string ss = str.substr( pos, pos2 == string::npos ? pos2 : ( pos2 - pos ) );
    string res = ss; //lang::UndecorateString( ss );
    if ( pos2 != string::npos && str[pos2] == ',' )
        ++pos2;
    pos = pos2;
    return res;
}

bool IsQuotedString( const string& str )
{
    return str.size() >= 2 && str[0] == '"' && str[str.size() - 1] == '"';
}

bool IsWord( const string& str )
{
    //bool bResult = str.empty() || isalpha( str[0] ) || str[0] == '_';
    bool bResult = true;
    for ( size_t i = 0; i < str.size() && bResult; i++ )
        bResult &= ( isalnum( str[i] ) != 0 ) || str[i] == '_';
    return bResult;
}



v_Token StringToTokens( const string& str )
{
    v_Token vt;
    Token t; // token currently is being filled
    string sPrev; // previous word
    size_t pos = 0;
    for ( string s; ; )
    {
         s = TokenItemNext( str, pos );
         bool bQuotedString = IsQuotedString( s );
         if ( bQuotedString )
             s = ctm::lang::UndecorateString( s );
         if ( bQuotedString || IsWord( s ) )
         {
             if ( !sPrev.empty() )
             {
                 if ( t.name.empty() )
                     throw runtime_error( "Property name expected." );
                 t.args.push_back( sPrev );
             }
             sPrev = s;
         }
         else if ( s == "=" )
         {
             if ( sPrev.empty() )
                 throw runtime_error( "Property name expected." );
             if ( !t.name.empty() )
                 vt.push_back( t );
             t = Token();
             t.name = ToUpper( sPrev );
             sPrev.clear();
         }
         else
             throw runtime_error( "Unknown character(s): " + s );
         if ( pos == string::npos )
         {
             if ( !t.name.empty() )
             {
                 if ( !sPrev.empty() )
                     t.args.push_back( sPrev );
                 vt.push_back( t );
             }
             else if ( !sPrev.empty() )
                 throw runtime_error( "Unknown record name: " + sPrev );
             break;
         }
    };

    return vt;
}

void RecordReaderHeader::XProcessRecord( CreateInstPtr& pNext, string& str )
{
    string buf = m_initBuf;
    pNext = 0;
    while ( !m_pStream->eof() && !pNext )
    {
        getline( *m_pStream, str );
	//	size_t n = m_pStream->tellg();
        size_t pos = str.find_first_not_of( " \t" );
        if ( pos == string::npos || str[pos] == '#' )
            continue;

        pNext = NewRecordStarted( str, pos );
        if ( !pNext )
            buf += " " + str;
    };
    v_Token vt = StringToTokens( buf );
    TecplotHeader header;
    for ( size_t i = 0; i < vt.size(); i++ )
    {
        if ( vt[i].name == "TITLE" && vt[i].args.size() )
            header.title = vt[i].args[0];
        else if ( vt[i].name == "VARIABLES" )
            header.vars = vt[i].args;
        else if ( vt[i].name == "FILETYPE" && vt[i].args.size() )
        {
            string type = ToUpper( vt[i].args[0] );
            if ( type == "FULL" )
                header.type = TECPLOT_FT_FULL;
            else if ( type == "GRID" )
                header.type = TECPLOT_FT_GRID;
            else if ( type == "SOLUTION" )
                header.type = TECPLOT_FT_SOLUTION;
            else throw runtime_error( "Unknown Tecplot file type: '" + type + "'" );
        }
        else
            throw runtime_error( "Unknown header property: '" + vt[i].name + "'" );
    };
    m_pTpR->SetHeader( header );
}

bool IsNumericString( const string& str )
{
    bool bIsNum = true;
    size_t pos = 0;
    while ( bIsNum && pos != string::npos )
    {
        pos = str.find_first_not_of( " \t", pos );
        if ( pos == string::npos )
            break;
        size_t pos2 = str.find_first_of( " \t", pos );
        string s = str.substr( pos, pos2 == string::npos ? string::npos : pos2 - pos );
        bIsNum &= ctm::lang::IsNumber( s.c_str() );
        pos = pos2;
    };
    return bIsNum;
}


struct ZoneInfoPriv : ZoneInfo
{
    bool m_bZoneTypeSet = false;
	XTecplotReaderASCII* m_pTpR;
	void ProcessToken( const Token& token );
	void OnTokenT( const Token& token );
	void OnTokenN( const Token& token );
	void OnTokenE( const Token& token );
	void OnTokenET( const Token& token );
	void OnTokenF( const Token& token );
	void OnTokenIJK( const Token& token );
	void OnTokenUnknown( const Token& token );
	void SetZoneType( bool bOrdered, const Token& token );
	void SetZoneTypeOrdered( const Token& token )
	{
		SetZoneType( true, token ); 
	}
	void SetZoneTypeFE( const Token& token )
	{
		SetZoneType( false, token );
	}
    void CalcBufferSizes();
};

void ZoneInfoPriv::ProcessToken( const Token& token )
{
	typedef void( ZoneInfoPriv::*OnTokenPtr )(const Token&);
	typedef map< string, OnTokenPtr > S2Ptr;
	static S2Ptr s2ptr;
	if ( s2ptr.empty() )
	{
		s2ptr["T"] = &ZoneInfoPriv::OnTokenT;
		s2ptr["N"] = &ZoneInfoPriv::OnTokenN;
		s2ptr["E"] = &ZoneInfoPriv::OnTokenE;
		s2ptr["ET"] = &ZoneInfoPriv::OnTokenET;
		s2ptr["F"] = &ZoneInfoPriv::OnTokenF;
		s2ptr["I"] = &ZoneInfoPriv::OnTokenIJK;
		s2ptr["J"] = &ZoneInfoPriv::OnTokenIJK;
		s2ptr["K"] = &ZoneInfoPriv::OnTokenIJK;
	};
	S2Ptr::const_iterator it = s2ptr.find( token.name );
	if ( it != s2ptr.end() )
		(this->*(it->second))(token);
	else
		OnTokenUnknown( token );
}


#define TEST_ARG( )\
if ( token.args.empty( ) || token.args[0].empty( ) )\
    throw runtime_error( token.name + " option requires argument. " );

#define TEST_ARG_INT( )\
if ( token.args.empty( ) || token.args[0].empty( ) || !ctm::lang::IsIntegerNumber( token.args[0].c_str( ) ) )\
    throw runtime_error( token.name + " option requires integer argument. " );


void ZoneInfoPriv::OnTokenT( const Token& token )
{
	if ( token.args.empty() )
        throw runtime_error( "T option requires argument. " );
	title = token.args[0];
}


void ZoneInfoPriv::OnTokenN( const Token& token )
{
	SetZoneTypeFE( token );
	TEST_ARG_INT();
	uNumNode = atoi( token.args[0].c_str() );
}

void ZoneInfoPriv::OnTokenE( const Token& token )
{
	SetZoneTypeFE( token );
	TEST_ARG_INT();
	uNumElem = atoi( token.args[0].c_str( ) );
}

void ZoneInfoPriv::OnTokenET( const Token& token )
{
	SetZoneTypeFE( token );
	TEST_ARG();
	const string& s = token.args[0];
	if ( s.compare( "TRIANGLE" ) == 0 )
		uElemType = TECPLOT_ET_TRI;
	else if ( s.compare( "QUADRILATERAL" ) == 0 )
		uElemType = TECPLOT_ET_QUAD;
	else if ( s.compare( "TETRAHEDRON" ) == 0 )
		uElemType = TECPLOT_ET_TET;
	else if ( s.compare( "BRICK" ) == 0 )
		uElemType = TECPLOT_ET_BRICK;
	else 
        throw runtime_error( "Invalid ET option value: '" + s + "'" );
}

void ZoneInfoPriv::OnTokenF( const Token& token )
{
	TEST_ARG();
	string s = ToUpper( token.args[0] );
	if ( s != "POINT" && s != "FEPOINT" && s != "BLOCK" && s != "FEBLOCK" )
        throw runtime_error( "F=\"" + s + "\": invalid option F argument. POINT, BLOCK, FEPOINT, FEBLOCK are only currently applicable." );
	bool bOrdered = s == "POINT" || s == "BLOCK";
	SetZoneType( bOrdered, token );
	bPoint = s == "POINT" || s == "FEPOINT";
}

void ZoneInfoPriv::OnTokenIJK( const Token& token )
{
	TEST_ARG_INT();
	SetZoneTypeOrdered( token );
	char c = token.name[0];
	ijk[c - 'I'] = atoi( token.args[0].c_str() );
	//uNumElem = (ijk[0] - 1)*(ijk[1] - 1)*(ijk[2] - 1);
	uNumNode = ijk[0] * ijk[1] * ijk[2];
    uElemType = ijk[0] < 2  ||  ijk[1] < 2   ||   ijk[2] < 2?  TECPLOT_ET_QUAD :   TECPLOT_ET_BRICK;
}

void ZoneInfoPriv::OnTokenUnknown( const Token& token )
{
}

#undef TEST_ARG
#undef TEST_ARG_INT

void ZoneInfoPriv::SetZoneType( bool bOrdered, const Token& token )
{
	if ( m_bZoneTypeSet && ( bOrdered ^ bIsOrdered ) )
	{
		string type = bIsOrdered ? string( "ORDERED" ) : "FINITE-ELEMENT";
		string prompt = "Current zone type (" + type + ") is incompatible with option '" + token.name;
		prompt += "=";
		for ( size_t i = 0; i < token.args.size(); i++ )
			prompt += "\"" + token.args[i] + "\" ";
		prompt += "'";
        throw runtime_error( prompt );
	};
	bIsOrdered = bOrdered;
	m_bZoneTypeSet = true;
}

void ZoneInfoPriv::CalcBufferSizes()
{
    assert( m_pTpR );
    size_t uVar = m_pTpR->GetHeader().vars.size();
    if ( bIsOrdered )
    {
        uBufSizeVar = uVar*ijk[0]*ijk[1]*ijk[2];
        uBufSizeCnt = 0;
    }
    else
    {
        uBufSizeVar = uVar*uNumNode;
        uBufSizeCnt = uNumElem*NodePerElem( uElemType );
    };
}

void RecordReaderZone::XProcessRecord( CreateInstPtr& pNext, string& str )
{
    string buf = m_initBuf;
    pNext = 0;
    bool bNumericString = false;
    size_t uPosVar = 0;
    while ( !m_pStream->eof() && !pNext )
    {
        uPosVar = m_pStream->tellg();
        getline( *m_pStream, str );
        size_t pos = str.find_first_not_of( " \t" );
        if ( pos == string::npos || str[pos] == '#' )
            continue;

        pNext = NewRecordStarted( str, pos );
        if ( !pNext )
        {
            if ( bNumericString = IsNumericString( str ) )
                break;
            buf += " " + str;
        };
    };
    size_t p = 0;
    TokenItemNext( buf, p ); // remove the word "zone"
    buf.erase( 0, p );
    v_Token vt = StringToTokens( buf );
	ZoneInfoPriv zi;
	zi.m_pTpR = m_pTpR;
	for ( size_t i = 0; i < vt.size(); i++ )
		zi.ProcessToken( vt[i] );
    zi.CalcBufferSizes();
	m_pTpR->AddZoneInfo( zi );
    ZonePosition zp( uPosVar );
    m_pTpR->AddZonePosition( zp );
	
    while ( !m_pStream->eof() && !pNext )
    {
        uPosVar = m_pStream->tellg();
        getline( *m_pStream, str );
        size_t pos = str.find_first_not_of( " \t" );
        if ( pos == string::npos || str[pos] == '#' )
            continue;
        pNext = NewRecordStarted( str, pos );
    };
}

void RecordReaderUnknown::XProcessRecord( CreateInstPtr& pNext, string& str )
{
    pNext = 0;
    while ( !m_pStream->eof() && !pNext )
    {
        getline( *m_pStream, str );
        size_t pos = str.find_first_not_of( " \t" );
        if ( pos == string::npos || str[pos] == '#' )
            continue;
        pNext = NewRecordStarted( str, pos );
    };
}


TecplotHeader XTecplotReaderASCII::GetHeader() const
{
    return m_header;
}


const ZoneInfo_v& XTecplotReaderASCII::GetZoneInfo() const
{
    return m_zoneInfo;
}

void XTecplotReaderASCII::GetZoneVariableValues( size_t nZone, double* pbuf ) const
{
    GetZoneVariableValuesT( nZone, pbuf );
}

void XTecplotReaderASCII::GetZoneVariableValues( size_t nZone, float* pbuf ) const
{
    GetZoneVariableValuesT( nZone, pbuf );
}

void XTecplotReaderASCII::GetZoneConnectivity( size_t nZone, unsigned int* pbuf ) const
{
    assert( m_pStream );
    const ZoneInfo_v& vzi = GetZoneInfo();
    if ( vzi.size() <= nZone )
        throw runtime_error( "XTecplotReaderASCII::GetZoneVariableValuesT: Zone index is out of range." );
    const ZoneInfo& zi = vzi[nZone];
    const ZonePosition& zp = ( GetZonePosition() )[nZone];
    if ( zp.uPosCnt == ~0 )
    {
        GetZoneVariableValuesT( nZone, ( float* )0 );
        ZonePosition zp2 = zp;
        zp2.uPosCnt = m_pStream->tellg();
        const_cast< XTecplotReaderASCII* >( this )->SetZonePosition( zp2, nZone );
    }
    else
    {
        m_pStream->clear();
        m_pStream->seekg( zp.uPosCnt );
    };
    size_t i = 0;
    for ( ; i < zi.uBufSizeCnt && !m_pStream->eof(); i++ )
	{
        *m_pStream >> pbuf[i];
		--pbuf[i]; // make it 0-based
	};
    if ( i < zi.uBufSizeCnt )
        throw runtime_error( "XTecplotReaderASCII::GetZoneConnectivity: Unexpected end of file while reading elements connectivities." );
}

void XTecplotReaderASCII::Open( const std::string& file )
{
    fstream* pf = new fstream;
    //cxx::MasterPtrRCX<istream> w = pf;
    shared_ptr<istream> w(pf);// = make_shared<fstream>();
    //fstream* pf = ( fstream* )w.get();
    pf->open( file.c_str(), ios_base::in|ios_base::binary );
    bool bGood = pf->good();
    bool bFail = pf->fail();
    bool bBad = pf->bad();
    if ( !pf->is_open() )
        throw runtime_error( "XTecplotReaderASCII::Open: Failed to open input file '" + file + "'" );
    Attach( *pf );
    m_wStream = w;
}

void XTecplotReaderASCII::Attach( std::istream& is )
{
    Close();
    m_pStream = &is;
    ReadFileStruct();
}

void XTecplotReaderASCII::Close()
{
    m_pStream = 0;
    m_wStream = 0;
    m_header = TecplotHeader();
    Clear();
}

void XTecplotReaderASCII::SetHeader( const TecplotHeader& header )
{
    Clear();
    m_header = header;
}

void XTecplotReaderASCII::AddZoneInfo( const ZoneInfo& zi )
{
	m_zoneInfo.push_back( zi );
}

void XTecplotReaderASCII::AddZonePosition( const ZonePosition& zp )
{
    m_zonePos.push_back( zp );
}

void XTecplotReaderASCII::SetZonePosition( const ZonePosition& zp, size_t nZone )
{
    assert( nZone < m_zonePos.size() );
    m_zonePos[nZone] = zp;
}

void XTecplotReaderASCII::ReadFileStruct()
{
    assert( m_pStream );
    //WIRR w = new RecordReaderHeader;
    WIRR w( new RecordReaderHeader );
    w->Init( this, "", *m_pStream );
    for ( ; w; w = WIRR( w->ProcessRecord() ) )
        ;
    m_pStream->clear();
}

void XTecplotReaderASCII::Clear()
{
    m_zoneInfo.clear();
    m_zonePos.clear();
}


template <typename T> void XTecplotReaderASCII::GetZoneVariableValuesT( size_t nZone, T* pbuf ) const
{
    assert( m_pStream );
    const ZoneInfo_v& vzi = GetZoneInfo();
    if ( vzi.size() <= nZone )
        throw runtime_error( "XTecplotReaderASCII::GetZoneVariableValuesT: Zone index is out of range." );
    const ZoneInfo& zi = vzi[nZone];
    size_t pos = ( GetZonePosition() )[nZone].uPosVars;
    m_pStream->clear();
    m_pStream->seekg( pos );
	//string str;
	//getline( *m_pStream, str );
    size_t i = 0;
    if ( pbuf )
    {
		for ( ; i < zi.uBufSizeVar && !m_pStream->eof(); i++ )
		{
			if ( i == 4802649 )
			{
				int k = 0;
			};
			*m_pStream >> pbuf[i];
		};
    }
    else
    {
        // Just to seek to connectivity section
        T x;
		for ( ; i < zi.uBufSizeVar && !m_pStream->eof(); i++ )
			*m_pStream >> x;
    };
    if ( i < zi.uBufSizeVar )
        throw runtime_error( "XTecplotReaderASCII::GetZoneVariableValuesT: Unexpected end of file while reading variables values." );
}

} // tecplot_rw namespace



TECPLOT_RW_API void tecplot_rw_create_reader( tecplot_rw::ITecplotReaderPtr* ptr, unsigned int type )
{
    assert( type == tecplot_rw::TECPLOT_READER_ASCII );
    *ptr = tecplot_rw::ITecplotReaderPtr( new tecplot_rw::XTecplotReaderASCII );
}

//}
//}


