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

#include "BinaryMeshReader.hpp"
#include "binary_io.hpp"
#include "foreach_byindex32.hpp"

using namespace std;

namespace s3dmm {

tecplot_rw::TecplotHeader BinaryMeshReader::GetHeader() const
{
    BOOST_ASSERT(m_s);
    tecplot_rw::TecplotHeader result;
    result.type = tecplot_rw::TECPLOT_FT_FULL;
    result.vars = m_vars;
    return result;
}

const tecplot_rw::ZoneInfo_v& BinaryMeshReader::GetZoneInfo() const
{
    BOOST_ASSERT(m_s);
    return m_zoneInfo;
}

void BinaryMeshReader::GetZoneVariableValues( size_t nZone, double* pbuf ) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

void BinaryMeshReader::GetZoneVariableValues( size_t nZone, float* pbuf ) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

template<class T>
void BinaryMeshReader::GetZoneVariableValuesPriv( size_t nZone, T* pbuf ) const
{
    BOOST_ASSERT(m_s);
    if (nZone >= m_zonePos.size())
        throw invalid_argument("Zone index is out of range");
    m_s->seekg(m_zoneDataPos[nZone]);
    m_s->read(reinterpret_cast<char*>(pbuf), static_cast<streamsize>(m_zoneInfo[nZone].uBufSizeVar*sizeof(T)));
}

void BinaryMeshReader::GetZoneConnectivity( size_t nZone, unsigned int* pbuf ) const
{
    BOOST_ASSERT(false);
    throw runtime_error("BinaryMeshReader::GetZoneConnectivity(): not implemented");
}

void BinaryMeshReader::Open( const std::string& file )
{
    Close();
    m_ownStream = make_shared<ifstream>(file, ios::binary);
    if (!m_ownStream->is_open())
        throw std::runtime_error(string("Failed to open input file '") + file + "'");
    Attach(*m_ownStream);
}

void BinaryMeshReader::Attach( std::istream& is )
{
    is.exceptions(istream::failbit);
    BinaryReader rd(is);
    string signature;
    rd >> signature;
    if (signature != "s3dmm_binary_79fbed93-3277-4ee4-accb-6f156987aa7a")
        throw runtime_error(string("Invalid or corrupted input file"));
    rd >> m_vars >> m_zonePos;
    m_zoneDataPos.resize(m_zonePos.size());

    m_zoneInfo.resize(m_zonePos.size());
    foreach_byindex32(izone, m_zonePos)
    {
        auto& zi = m_zoneInfo[izone];
        zi = tecplot_rw::ZoneInfo();
        // zi.title =
        zi.bIsOrdered = true;
        zi.bPoint = false;
        BinaryReader rd(is);
        is.seekg(static_cast<streamoff>(m_zonePos[izone]));
        auto ijk = rd.read<vector<size_t>>();
        m_zoneDataPos[izone] = is.tellg();
        auto dim = ijk.size();
        BOOST_ASSERT(dim >= 1 && dim <= 3);
        ijk.resize(3, 1);
        zi.ijk = ijk;
        zi.uNumNode = ijk[0] * ijk[1] * ijk[2];
        zi.uNumElem = 0;
        BOOST_ASSERT(dim >= 2);  // BUGgy: no support for 1D
        zi.uElemType = dim < 3? tecplot_rw::TECPLOT_ET_QUAD: tecplot_rw::TECPLOT_ET_BRICK;
        zi.uBufSizeVar = m_vars.size() * zi.uNumNode;
        zi.uBufSizeCnt = 0;
    }
    m_s = &is;
}

void BinaryMeshReader::Close()
{
    m_s = nullptr;
    m_ownStream.reset();
    m_vars.clear();
    m_zonePos.clear();
    m_zoneDataPos.clear();
    m_zoneInfo.clear();
}

} // s3dmm
