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

#ifndef TECPLOT_READER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22
#define TECPLOT_READER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22

#include "tecplot_rw.h"
#include <memory>
#include <vector>

namespace tecplot_rw {


//typedef float TecplotFloat;

enum {
    // TecplotHeader::type values
    TECPLOT_FT_FULL,
    TECPLOT_FT_GRID,
    TECPLOT_FT_SOLUTION,

    // ZoneInfo::uElemType values
    TECPLOT_ET_TRI = 0,
    TECPLOT_ET_QUAD,
    TECPLOT_ET_TET,
    TECPLOT_ET_BRICK
};

// Tecplot ASCII file header info
struct TecplotHeader
{
    std::string title; // file title (see TITLE Tecplot ASCII format property)
    unsigned int type; // file type, one of TECPLOT_FT_ values (see FILETYPE Tecplot ASCII format property)
    std::vector< std::string > vars; // variables names (see VARIABLES Tecplot ASCII format property)
    TecplotHeader() { type = TECPLOT_FT_FULL; }
};

// Tecplot ASCII file zone info
struct ZoneInfo
{
    std::string  title; // zone title (T record)
    bool         bIsOrdered; // zone type. Ordered if true, finite-element otherwise
    bool         bPoint;     // zone format. Point if true, block otherwise
    std::vector<size_t> ijk;// IJK dimensions. Has sense for ordered zones only
    size_t       uNumNode;   // node number
    size_t       uNumElem;   // FE number (always 0 for ordered zone)
    unsigned int uElemType; // FE type. One of TECPLOT_ET_ values. Has sense for finite-element zones only

    size_t       uBufSizeVar; // buffer size for all variable values
    size_t       uBufSizeCnt; // buffer size for element connectivities (0 for ordered zone)
    ZoneInfo() : bIsOrdered( true ), bPoint( true ), ijk( 3, 1 ), uNumNode( 0 ), uNumElem( 0 ),
        uElemType( TECPLOT_ET_BRICK ), uBufSizeCnt( 0 ), uBufSizeVar( 0 ) {}
};
typedef std::vector< ZoneInfo > ZoneInfo_v;


inline size_t NodePerElem( unsigned int uType )
{
    static size_t u[] = {3,4,4,8};
    return u[uType];
}


struct ITecplotReader
{
    virtual ~ITecplotReader() {}
    virtual TecplotHeader GetHeader() const = 0;
    virtual const ZoneInfo_v& GetZoneInfo() const = 0;
    virtual void GetZoneVariableValues( size_t nZone, double* pbuf ) const = 0;
    virtual void GetZoneVariableValues( size_t nZone, float* pbuf ) const = 0;
    virtual void GetZoneConnectivity( size_t nZone, unsigned int* pbuf ) const = 0;

    virtual void Open( const std::string& file ) = 0;
    virtual void Attach( std::istream& is ) = 0;
    virtual void Close() = 0;
};

typedef std::shared_ptr<ITecplotReader> ITecplotReaderPtr;

enum TECPLOT_READER_TYPE {
    TECPLOT_READER_ASCII
};




} // tecplot_rw



#ifdef __cplusplus
extern "C"
    {
#endif // __cplusplus

    TECPLOT_RW_API void tecplot_rw_create_reader( tecplot_rw::ITecplotReaderPtr* ptr, unsigned int type );

#ifdef __cplusplus
    }
#endif // __cplusplus



namespace tecplot_rw {


inline ITecplotReaderPtr CreateReader( unsigned int type = TECPLOT_READER_ASCII ) {
    ITecplotReaderPtr ptr;
    tecplot_rw_create_reader( &ptr, type );
    return ptr;
}


} // tecplot_rw

#endif // TECPLOT_READER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22
