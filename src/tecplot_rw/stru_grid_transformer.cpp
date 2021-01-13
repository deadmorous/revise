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


#include "tecplot_rw/stru_grid_transformer.h"
#include "Vec.hpp"

using namespace std;

namespace tecplot_rw {


class XStruGridTransformer :
    public IStruGridTransformer
{
public:
    virtual void Transform( std::vector<double>& data, const s3dmm::Vec3u& sizes,
                            s3dmm::Vec3u& resultSize, unsigned int stride, bool bEvalResultSize );
    virtual bool DummyMode( bool bDummyMode ){
        swap( m_bDummyMode, bDummyMode );
        return bDummyMode;
    }
    virtual bool DummyMode() const {
        return m_bDummyMode;
    }
    virtual void SetRelativeLOD( double lod ) {
        m_dRelativeLOD = lod;
    }
    virtual double GetRelativeLOD() const {
        return m_dRelativeLOD;
    }
    XStruGridTransformer() : m_dRelativeLOD( 1.0 ) {}
private:
    bool m_bDummyMode = false;
    double m_dRelativeLOD;
};

void XStruGridTransformer::Transform(std::vector<double>& data, const s3dmm::Vec3u &sizes,
                                      s3dmm::Vec3u &resultSize, unsigned int stride, bool bEvalResultSize)
{
    if ( m_bDummyMode )
        return;
    // TODO
}

} // tecplot_rw

TECPLOT_RW_API void tecplot_rw_create_stru_grid_transformer( tecplot_rw::IStruGridTransformerPtr* ptr )
{
    *ptr = tecplot_rw::IStruGridTransformerPtr( new tecplot_rw::XStruGridTransformer );
}




