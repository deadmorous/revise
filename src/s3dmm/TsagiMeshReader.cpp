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

#include "TsagiMeshReader.h"

using namespace std;

namespace s3dmm {

tecplot_rw::TecplotHeader TsagiMeshReader::GetHeader() const
{
    tecplot_rw::TecplotHeader result;
    result.type = tecplot_rw::TECPLOT_FT_FULL;
    result.title = "Tsagi_Mesh";
    result.vars = {"x", "y", "z", "DP_DT", "U", "GRAD_RO", "Qcr"};
    return result;
}

const tecplot_rw::ZoneInfo_v &TsagiMeshReader::GetZoneInfo() const
{
    return m_zoneInfo;
}

void TsagiMeshReader::GetZoneVariableValues(size_t nZone, double *pbuf) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

void TsagiMeshReader::GetZoneVariableValues(size_t nZone, float *pbuf) const
{
    GetZoneVariableValuesPriv( nZone, pbuf );
}

void TsagiMeshReader::GetZoneConnectivity(size_t /*nZone*/, unsigned int */*pbuf*/) const
{
    throw std::runtime_error(string("GetConnectivity is not implemented"));
}

void TsagiMeshReader::Open(const string &file)
{
    string grid_fname, field_fname;

    ifstream is(file);
    is.exceptions(istream::failbit);
    getline(is,grid_fname);
    getline(is,field_fname);
    is >> m_timestep;
    is.close();

    m_gridStream=make_shared<ifstream>(grid_fname, ios::binary);
    if (!m_gridStream->is_open())
        throw std::runtime_error(string("Failed to open input file '") + grid_fname + "'");
    m_fieldStream=make_shared<ifstream>(field_fname, ios::binary);
    if (!m_fieldStream->is_open())
        throw std::runtime_error(string("Failed to open input file '") + grid_fname + "'");
    m_gridStream->exceptions(istream::failbit);
    m_fieldStream->exceptions(istream::failbit);
    Attach(*m_gridStream);
}

void TsagiMeshReader::Attach(istream &is)
{
    //int and float must be 4 bytes long
    unsigned int nblocks=0;
    is.read((char*)&nblocks,sizeof(int)); //number of zones in one timestep
    m_zoneInfo.resize(nblocks);

    m_gridcoords.resize(nblocks);

    vector <size_t> block_sizes;  //block array sizes in points; will be used to read solution
    m_total_grid_size=0;  //grid size in points(for info only)

    for(size_t i=0;i<nblocks;++i)
    {
        auto& zi = m_zoneInfo[i];
        zi = tecplot_rw::ZoneInfo();
        zi.bIsOrdered = true;
        zi.bPoint = false;

        zi.title=string("Zone_")+std::to_string(i);

        int ni=0,nj=0,nk=0;
        //reading grid blocks dimensions
        is.read((char*)&ni,sizeof(int));
        is.read((char*)&nj,sizeof(int));
        is.read((char*)&nk,sizeof(int));

        zi.ijk[0]=ni;
        zi.ijk[1]=nj;
        zi.ijk[2]=nk;

        zi.uNumNode=ni*nj*nk;

        m_total_grid_size+=zi.uNumNode;

        zi.uNumElem = 0;
        zi.uElemType = tecplot_rw::TECPLOT_ET_BRICK;
        zi.uBufSizeVar = (3+m_numvars)*sizeof(float) * zi.uNumNode;// we have 3 float coords and 4 float vars
        zi.uBufSizeCnt = 0;

        ///--------

        block_sizes.push_back(ni*nj*nk);

        size_t float_arr_size=3*ni*nj*nk;
        m_gridcoords[i].resize(float_arr_size);
        is.read((char*)m_gridcoords[i].data(),float_arr_size*sizeof(float));

    }
}

void TsagiMeshReader::Close()
{
    m_gridStream.reset();
    m_fieldStream.reset();
    m_gridcoords.clear();
    m_zoneInfo.clear();
    m_total_grid_size=0;
}

template<class T>
void TsagiMeshReader::GetZoneVariableValuesPriv(size_t nZone, T *pbuf) const
{
    size_t varsize=m_zoneInfo[nZone].uNumNode*m_numvars;
    size_t hdrsize_bytes=8; // 2 ints: Index and LEVTIM

    size_t timestep_size_bytes=m_total_grid_size*m_numvars*sizeof(float)+hdrsize_bytes;

    size_t fieldoffset=0;
    for(size_t i=0;i<nZone;++i) fieldoffset+=m_zoneInfo[i].uNumNode*m_numvars*sizeof(float);

    m_fieldStream->seekg(timestep_size_bytes*m_timestep+fieldoffset+hdrsize_bytes);
    vector<float> vardata(varsize);
    m_fieldStream->read((char*)vardata.data(),varsize*sizeof(float));

    size_t step=3+m_numvars;
    for(size_t i=0;i<m_zoneInfo[nZone].uNumNode;++i)
    {
        auto coordstart=m_gridcoords[nZone].begin()+i*3;
        copy(coordstart,coordstart+3,pbuf+i*step);
        auto varstart=vardata.begin()+i*m_numvars;
        copy(varstart,varstart+m_numvars,pbuf+i*step+3);
    }
}

}
