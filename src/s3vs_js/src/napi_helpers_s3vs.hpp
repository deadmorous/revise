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

#include "napi_helpers.hpp"

#include "s3vs/VsControllerInterface.hpp"

template<>
struct Ntype<s3vs::ClippingPlane>
{
    using type = Napi::Object;

    static Napi::Object toNvalue(Napi::Env env, const s3vs::ClippingPlane& t)
    {
        auto result = Napi::Object::New(env);
        result["pos"] = makeArrayObj(env, t.pos);
        result["normal"] = makeArrayObj(env, t.normal);
        return result;
    }

    static s3vs::ClippingPlane toTInitializer(const Napi::Object& n)
    {
        s3vs::ClippingPlane result;
        result.pos = readRequiredFixedLengthArrayProperty<3, s3dmm::real_type>(n, "pos");
        result.normal = readRequiredFixedLengthArrayProperty<3, s3dmm::real_type>(n, "normal");
        return result;
    }
};
