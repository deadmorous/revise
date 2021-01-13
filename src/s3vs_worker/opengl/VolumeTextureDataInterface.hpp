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

#include <vlGraphics/Texture.hpp>
#include "s3dmm/BlockTreeFieldService.hpp"
#include "silver_bullets/factory.hpp"

class VolumeTextureDataInterface :
    public silver_bullets::Factory<VolumeTextureDataInterface>
{
public:
    using BlockId = s3dmm::detail::TreeBlockId<3>;
    virtual ~VolumeTextureDataInterface() = default;
    virtual void setFieldService(s3dmm::BlockTreeFieldService<3> *fieldService) = 0;
    virtual void setField(
            unsigned int fieldIndex, unsigned int timeFrame,
            const BlockId& m_subtreeRoot) = 0;
    virtual vl::Texture *fieldTexture() const = 0;
    virtual vl::Texture *alphaTexture() const = 0;
    virtual unsigned int depth() const = 0;
    virtual s3dmm::Vec2<s3dmm::dfield_real> fieldRange() const = 0;

    unsigned int textureEdgeSize() const {
        return (1 << depth()) + 1;
    }
};
