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

#include "BoundingBox.hpp"

#include "silver_bullets/iterate_struct/iterate_struct.hpp"
#include "silver_bullets/iterate_struct/PlainRepresentation.hpp"

namespace s3dmm {

template <unsigned int N, class T>
struct BoundingBoxPlainRepresentation
{
    using vector_type = typename s3dmm::BoundingBox<N,T>::vector_type;
    vector_type min;
    vector_type max;

    static BoundingBoxPlainRepresentation<N,T> toPlain(const s3dmm::BoundingBox<N,T>& x)
    {
        BoundingBoxPlainRepresentation<N,T> result;
        if (x.empty()) {
            s3dmm::ScalarOrMultiIndex<N,T>::each(result.min, [](T& e) { e = 1; });
            s3dmm::ScalarOrMultiIndex<N,T>::each(result.max, [](T& e) { e = 0; });
        }
        else {
            result.min = x.min();
            result.max = x.max();
        }
        return result;
    }

    static s3dmm::BoundingBox<N,T> fromPlain(const BoundingBoxPlainRepresentation<N,T>& x)
    {
        s3dmm::BoundingBox<N,T> result;
        if (s3dmm::ScalarOrMultiIndex<N,T>::element(x.min, 0) <= s3dmm::ScalarOrMultiIndex<N,T>::element(x.max, 0))
            result << x.min << x.max;
        return result;
    }
};

} // s3dmm

SILVER_BULLETS_DESCRIBE_TEMPLATE_STRUCTURE_FIELDS(
    ((unsigned int, N), (class, T)),
    s3dmm::BoundingBoxPlainRepresentation, min, max)

template <unsigned int N, class T>
struct silver_bullets::iterate_struct::PlainRepresentation<s3dmm::BoundingBox<N,T>> {
    using type = s3dmm::BoundingBoxPlainRepresentation<N, T>;
};
