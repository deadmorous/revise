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

#include "MeshElementType.hpp"
#include "Vec.hpp"

#include <boost/assert.hpp>

namespace s3dmm {

enum class ElementSubtype
{
    Triangle,
    Quad,
    Tetrahedron,
    Pyramid,
    Prism,
    Hexahedron
};

template<MeshElementType elementType> struct MeshElementTraits;

template<> struct MeshElementTraits<MeshElementType::Triangle>
{
    static constexpr const MeshElementType ElementType = MeshElementType::Triangle;
    static constexpr const unsigned int SpaceDimension = 2;
    static constexpr const unsigned int ElementNodeCount = 3;
    static constexpr const unsigned int ElementFaceCount = 3;
    static constexpr const unsigned int ElementFaceSize = 2;
    using Face = MultiIndex<ElementFaceSize, unsigned int>;
    using Node = Vec<SpaceDimension, real_type>;
    static ElementSubtype elementSubtype(const unsigned int */*elementNodeNumbers*/) {
        return ElementSubtype::Triangle;
    }
    static constexpr unsigned int elementFaceCount(ElementSubtype) {
        return ElementFaceCount;
    }
    static Face localFaceIndices(unsigned int iface)
    {
        static Face faces[ElementFaceCount] = { {0, 1}, {1, 2}, {2, 0} };
        BOOST_ASSERT(iface < ElementFaceCount);
        return faces[iface];
    }
    static Face localFaceIndices(unsigned int iface, ElementSubtype) {
        return localFaceIndices(iface);
    }
    static constexpr unsigned int subtypeIndex(ElementSubtype)
    {
        return ~0u;
    }
    static constexpr unsigned int subtypeSimplexCount(unsigned int)
    {
        return 0;
    }
    static MultiIndex<3, unsigned int> subtypeLocalSimplexIndices(
        unsigned int, unsigned int)
    {
        BOOST_ASSERT(false);
        return {~0u, ~0u, ~0u};
    }
};

template<> struct MeshElementTraits<MeshElementType::Quad>
{
    static constexpr const MeshElementType ElementType = MeshElementType::Quad;
    static constexpr const unsigned int SpaceDimension = 2;
    static constexpr const unsigned int ElementNodeCount = 4;
    static constexpr const unsigned int ElementFaceCount = 4;
    static constexpr const unsigned int ElementFaceSize = 2;
    using Face = MultiIndex<ElementFaceSize, unsigned int>;
    using Node = Vec<SpaceDimension, real_type>;
    static ElementSubtype elementSubtype(const unsigned int *elementNodeNumbers) {
        if (elementNodeNumbers[3] != ~0u)
            return ElementSubtype::Quad;
        else
            return ElementSubtype::Triangle;
    }
    static constexpr unsigned int elementFaceCount(ElementSubtype elementSubtype) {
        switch (elementSubtype) {
            case ElementSubtype::Triangle:
                return 3;
            case ElementSubtype::Quad:
                return 4;
            default:
                BOOST_ASSERT(false);
                return 0;
        }
    }
    static Face localFaceIndices(unsigned int iface)
    {
        static Face faces[ElementFaceCount] = { {3, 0}, {1, 2}, {0, 1}, {2, 3} };
        BOOST_ASSERT(iface < ElementFaceCount);
        return faces[iface];
    }
    static Face localFaceIndices(unsigned int iface, ElementSubtype subtype) {
        static Face faces[2][ElementFaceCount] = {
            {{0, 1}, {1, 2}, {2, 0}, {~0u,~0u}},
            {{3, 0}, {1, 2}, {0, 1}, {2, 3} }
        };
        BOOST_ASSERT(iface < elementFaceCount(subtype));
        return faces[subtypeIndex(subtype)][iface];
    }
    static constexpr unsigned int subtypeIndex(ElementSubtype subtype)
    {
        auto result = static_cast<unsigned int>(subtype) -
                      static_cast<unsigned int>(ElementSubtype::Triangle);
        BOOST_ASSERT(result < 2);
        return result;
    }
    static constexpr unsigned int subtypeSimplexCount(unsigned int subtypeIndex)
    {
        constexpr unsigned int subtypeSimplexCounts[] = { 1, 2 };
        BOOST_ASSERT(subtypeIndex < 2);
        return subtypeSimplexCounts[subtypeIndex];
    }
    static MultiIndex<3, unsigned int> subtypeLocalSimplexIndices(
        unsigned int subtypeIndex, unsigned int simplexIndex)
    {
        static MultiIndex<3, unsigned int> indices[4][5] = {
            {{0, 1, 2}},
            {{0, 1, 2}, {0, 2, 3}}
        };
        BOOST_ASSERT(simplexIndex < subtypeSimplexCount(subtypeIndex));
        return indices[subtypeIndex][simplexIndex];
    }
};

template<> struct MeshElementTraits<MeshElementType::Tetrahedron>
{
    static constexpr const MeshElementType ElementType = MeshElementType::Tetrahedron;
    static constexpr const unsigned int SpaceDimension = 3;
    static constexpr const unsigned int ElementNodeCount = 4;
    static constexpr const unsigned int ElementFaceCount = 4;
    static constexpr const unsigned int ElementFaceSize = 3;
    using Face = MultiIndex<ElementFaceSize, unsigned int>;
    using Node = Vec<SpaceDimension, real_type>;
    static ElementSubtype elementSubtype(const unsigned int */*elementNodeNumbers*/) {
        return ElementSubtype::Tetrahedron;
    }
    static constexpr unsigned int elementFaceCount(ElementSubtype) {
        return ElementFaceCount;
    }
    static Face localFaceIndices(unsigned int iface)
    {
        // TODO: Check face orientation
        static Face faces[ElementFaceCount] = { {0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3} };
        BOOST_ASSERT(iface < ElementFaceCount);
        return faces[iface];
    }
    static Face localFaceIndices(unsigned int iface, ElementSubtype) {
        return localFaceIndices(iface);
    }
    static constexpr unsigned int subtypeIndex(ElementSubtype)
    {
        return ~0u;
    }
    static constexpr unsigned int subtypeSimplexCount(unsigned int)
    {
        return 0;
    }
    static MultiIndex<4, unsigned int> subtypeLocalSimplexIndices(
        unsigned int, unsigned int)
    {
        BOOST_ASSERT(false);
        return {~0u, ~0u, ~0u, ~0u};
    }
};

template<> struct MeshElementTraits<MeshElementType::Hexahedron>
{
    static constexpr const MeshElementType ElementType = MeshElementType::Hexahedron;
    static constexpr const unsigned int SpaceDimension = 3;
    static constexpr const unsigned int ElementNodeCount = 8;
    static constexpr const unsigned int ElementFaceCount = 6;
    static constexpr const unsigned int ElementFaceSize = 4;
    using Face = MultiIndex<ElementFaceSize, unsigned int>;
    using Node = Vec<SpaceDimension, real_type>;
    static ElementSubtype elementSubtype(const unsigned int *elementNodeNumbers)
    {
        if (elementNodeNumbers[7] != ~0u)
            return ElementSubtype::Hexahedron;
        else if (elementNodeNumbers[5] != ~0u)
            return ElementSubtype::Prism;
        else if (elementNodeNumbers[4] != ~0u)
            return ElementSubtype::Pyramid;
        else
            return ElementSubtype::Tetrahedron;
    }
    static constexpr unsigned int elementFaceCount(ElementSubtype elementSubtype)
    {
        switch (elementSubtype) {
            case ElementSubtype::Tetrahedron:
                return 4;
            case ElementSubtype::Pyramid:
                return 5;
            case ElementSubtype::Prism:
                return 5;
            case ElementSubtype::Hexahedron:
                return 6;
            default:
                BOOST_ASSERT(false);
                return 0;
        }
    }
    static Face localFaceIndices(unsigned int iface)
    {
        static Face faces[ElementFaceCount] = {
            {0, 4, 7, 3}, {1, 2, 6, 5}, {0, 1, 5, 4}, {3, 7, 6, 2}, {0, 3, 2, 1}, {4, 5, 6, 7}
        };
        BOOST_ASSERT(iface < ElementFaceCount);
        return faces[iface];
    }
    static Face localFaceIndices(unsigned int iface, ElementSubtype subtype)
    {
        static Face faces[4][ElementFaceCount] = {
            {{0, 2, 1, ~0u}, {0, 1, 3, ~0u}, {1, 2, 3, ~0u}, {2, 0, 3, ~0u}, {~0u,~0u,~0u,~0u}, {~0u,~0u,~0u,~0u}},
            {{0, 3, 2,  1 }, {0, 1, 4, ~0u}, {1, 2, 4, ~0u}, {2, 3, 4, ~0u}, { 3, 0, 4, ~0u}, {~0u,~0u,~0u,~0u}},
            {{0, 2, 1, ~0u}, {3, 4, 5, ~0u}, {0, 1, 4,  3 }, {1, 2, 5,  4 }, { 2, 0, 3,  5 }, {~0u,~0u,~0u,~0u}},
            {{0, 4, 7,  3 }, {1, 2, 6,  5 }, {0, 1, 5,  4 }, {3, 7, 6,  2 }, { 0, 3, 2,  1 }, { 4, 5, 6, 7}}
        };
        BOOST_ASSERT(iface < elementFaceCount(subtype));
        return faces[subtypeIndex(subtype)][iface];
    }
    static constexpr unsigned int subtypeIndex(ElementSubtype subtype)
    {
        auto result = static_cast<unsigned int>(subtype) -
                      static_cast<unsigned int>(ElementSubtype::Tetrahedron);
        BOOST_ASSERT(result < 4);
        return result;
    }
    static constexpr unsigned int subtypeSimplexCount(unsigned int subtypeIndex)
    {
        constexpr unsigned int subtypeSimplexCounts[] = { 1, 2, 3, 5 };
        BOOST_ASSERT(subtypeIndex < 4);
        return subtypeSimplexCounts[subtypeIndex];
    }
    static MultiIndex<4, unsigned int> subtypeLocalSimplexIndices(
        unsigned int subtypeIndex, unsigned int simplexIndex)
    {
        static MultiIndex<4, unsigned int> indices[4][5] = {
            {{0, 1, 2, 3}},
            {{0, 1, 2, 4}, {0, 2, 3, 4}},
            {{0, 1, 2, 3}, {1, 4, 5, 3}, {1, 5, 2, 3}},
            {{1, 4, 3, 0}, {1, 3, 6, 2}, {1, 6, 4, 5}, {3, 4, 6, 7}, {1, 3, 4, 6}}
        };
        BOOST_ASSERT(simplexIndex < subtypeSimplexCount(subtypeIndex));
        return indices[subtypeIndex][simplexIndex];
    }
};

template <MeshElementType elementType>
using MeshElementFace = typename MeshElementTraits<elementType>::Face;

template <MeshElementType elementType>
using MeshElementNode = typename MeshElementTraits<elementType>::Node;



template<MeshElementType elementType, ElementSubtype subtype>
struct is_element_subtype : std::false_type {};

template<>
struct is_element_subtype<MeshElementType::Quad, ElementSubtype::Triangle> : std::true_type {};

template<>
struct is_element_subtype<MeshElementType::Hexahedron, ElementSubtype::Tetrahedron> : std::true_type {};

template<>
struct is_element_subtype<MeshElementType::Hexahedron, ElementSubtype::Pyramid> : std::true_type {};

template<>
struct is_element_subtype<MeshElementType::Hexahedron, ElementSubtype::Prism> : std::true_type {};

template<MeshElementType elementType, ElementSubtype subtype>
constexpr bool is_element_subtype_v = is_element_subtype<elementType, subtype>::value;

template<unsigned int N> struct simplex_element_type;

template <>
struct simplex_element_type<2> {
    static constexpr auto value = MeshElementType::Triangle;
};

template <>
struct simplex_element_type<3> {
    static constexpr auto value = MeshElementType::Tetrahedron;
};

template<unsigned int N>
constexpr auto simplex_element_type_v = simplex_element_type<N>::value;

} // s3dmm
