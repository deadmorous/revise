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

#include "tecplot_rw/tecplot_reader.h"
#include <fstream>

namespace s3dmm {

class BinaryMeshReader : public tecplot_rw::ITecplotReader
{
public:
    tecplot_rw::TecplotHeader GetHeader() const override;
    const tecplot_rw::ZoneInfo_v& GetZoneInfo() const override;
    void GetZoneVariableValues( size_t nZone, double* pbuf ) const override;
    void GetZoneVariableValues( size_t nZone, float* pbuf ) const override;
    void GetZoneConnectivity( size_t nZone, unsigned int* pbuf ) const override;

    void Open( const std::string& file ) override;
    void Attach( std::istream& is ) override;
    void Close()  override;

private:
    std::istream *m_s;
    std::shared_ptr<std::ifstream> m_ownStream;
    std::vector<std::string> m_vars;
    std::vector<std::size_t> m_zonePos;
    std::vector<std::streamoff> m_zoneDataPos;
    tecplot_rw::ZoneInfo_v m_zoneInfo;

    template<class T>
    void GetZoneVariableValuesPriv( size_t nZone, T* pbuf ) const;

};

} // s3dmm
