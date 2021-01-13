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

#ifndef STRU_GRID_TRANSFORMER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22
#define STRU_GRID_TRANSFORMER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22

#include "tecplot_rw.h"
#include "Vec.hpp"
#include <memory>
#include <vector>

namespace tecplot_rw {


struct IStruGridTransformer
{
    // Changes data in-place, the size of the vector may change. If bEvalResultSize==true
    // the resulting grid size resultSize is unknown and is to be calculated.
    // Otherwise resultSize is treated as given
    virtual void Transform( std::vector<double>& data, const s3dmm::Vec3u& sizes,
                            s3dmm::Vec3u& resultSize, unsigned int stride, bool bEvalResultSize ) = 0;

    virtual bool DummyMode( bool bDummyMode ) = 0;
    virtual bool DummyMode() const = 0;

    virtual void SetRelativeLOD( double lod ) = 0;
    virtual double GetRelativeLOD() const = 0;
};

typedef std::shared_ptr<IStruGridTransformer> IStruGridTransformerPtr;


} // tecplot_rw

#ifdef __cplusplus
extern "C"
    {
#endif // __cplusplus

    TECPLOT_RW_API void tecplot_rw_create_stru_grid_transformer( tecplot_rw::IStruGridTransformerPtr* ptr );

#ifdef __cplusplus
    }
#endif // __cplusplus



namespace tecplot_rw {

namespace stru_grid_transformer {

inline IStruGridTransformerPtr Create( ) {
    IStruGridTransformerPtr ptr;
    tecplot_rw_create_stru_grid_transformer( &ptr );
    return ptr;
}

} // stru_grid_transformer

} // tecplot_rw

#endif // STRU_GRID_TRANSFORMER_H_EAE2A2C9_6FD1_4e2e_AF16_3375B3F2FC22
