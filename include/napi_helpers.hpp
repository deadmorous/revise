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

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

#include <napi.h>

#include "silver_bullets/enum_names.hpp"
#include "MultiIndex.hpp"
#include "silver_bullets/sync/SyncAccessor.hpp"

#include <boost/lexical_cast.hpp>

template<class T, class = void> struct Ntype;
template<class T> using Ntype_t = typename Ntype<T>::type;

template<>
struct Ntype<bool>
{
    using type = Napi::Boolean;
    static type coerce(const Napi::Value& n) {
        return n.ToBoolean();
    }
    static Napi::Boolean toNvalue(Napi::Env env, bool t) {
        return Napi::Boolean::New(env, t);
    }
    static bool toTInitializer(const Napi::Boolean& n) {
        return n;
    }
};

template<class T>
struct Ntype<T, std::enable_if_t<std::is_arithmetic_v<T>, void>>
{
    using type = Napi::Number;
    static type coerce(const Napi::Value& n) {
        return n.ToNumber();
    }
    static Napi::Number toNvalue(Napi::Env env, T t) {
        return Napi::Number::New(env, t);
    }
    static T toTInitializer(const Napi::Number& n) {
        return n;
    }
};

template<>
struct Ntype<std::string>
{
    using type = Napi::String;
    static type coerce(const Napi::Value& n) {
        return n.ToString();
    }
    static Napi::String toNvalue(Napi::Env env, const std::string& t) {
        return Napi::String::New(env, t);
    }
    static std::string toTInitializer(const Napi::String& n) {
        return n;
    }
};

template<class T>
struct Ntype<T, std::enable_if_t<std::is_enum_v<T>, void>>
{
    using type = Napi::String;
    static type coerce(const Napi::Value& n) {
        return n.ToString();
    }
    static Napi::String toNvalue(Napi::Env env, T t) {
        return Napi::String::New(env, silver_bullets::enum_item_name(t));
    }
    static T toTInitializer(const Napi::String& n) {
        return silver_bullets::enum_item_value<T>(n);
    }
};

template<class T>
inline Ntype_t<T> toNvalue(Napi::Env env, const T& t) {
    return Ntype<T>::toNvalue(env, t);
};

template<class T>
inline T toTInitializer(const Ntype_t<T>& n) {
    return Ntype<T>::toTInitializer(n);
}



template <class T>
inline Napi::Array makeArrayObj(Napi::Env env, const T *elements, std::size_t count)
{
    auto result = Napi::Array::New(env, count);
    for (std::size_t i=0; i<count; ++i)
        result[i] = toNvalue(env, elements[i]);
    return result;
}

template <unsigned int N, class T>
inline Napi::Array makeArrayObj(Napi::Env env, const s3dmm::MultiIndex<N,T>& v) {
    return makeArrayObj(env, v.data(), N);
}

template <class T>
inline Napi::Array makeArrayObj(Napi::Env env, const std::vector<T>& v) {
    return makeArrayObj(env, v.data(), v.size());
}

template <unsigned int N, class T>
inline void readFixedLengthArrayObj(T *dst, const Napi::Value& val)
{
    if (!val.IsArray())
        throw std::runtime_error("An array is expected");
    auto a = val.As<Napi::Array>();
    if (a.Length() != N)
        throw std::runtime_error(
                std::string("An array of length ") + boost::lexical_cast<std::string>(N) + " is expected");
    for (std::size_t i=0; i<N; ++i) {
        auto ai = a.Get(i);
        dst[i] = toTInitializer<T>(ai.As<Ntype_t<T>>());
    }
}

template <unsigned int N, class T>
inline s3dmm::MultiIndex<N,T> readFixedLengthArrayObj(const Napi::Value& val)
{
    s3dmm::MultiIndex<N,T> result;
    readFixedLengthArrayObj<N, T>(result.data(), val);
    return result;
}

template <unsigned int N, class T>
inline auto fixedLengthArrayObjReader() {
    return [](const Napi::Value& val) {
        return readFixedLengthArrayObj<N, T>(val);
    };
}

template <class T>
inline std::vector<T> readArrayObj(const Napi::Value& val)
{
    if (!val.IsArray())
        throw std::runtime_error("An array is expected");
    auto a = val.As<Napi::Array>();
    auto length = a.Length();
    std::vector<T> result(length);
    for (std::size_t i=0; i<length; ++i) {
        auto ai = a.Get(i);
        result[i] = toTInitializer<T>(ai.As<Ntype_t<T>>());
    }
    return result;
}

template <unsigned int N, class T>
inline auto arrayObjReader() {
    return [](const Napi::Value& val) {
        return readArrayObj<T>(val);
    };
}

template <class ValueParser>
inline auto readRequiredProperty(Napi::Object o, const char *name, ValueParser valueParser)
{
    auto prop = o.Get(name);
    if (prop.IsUndefined())
        throw std::runtime_error(
                std::string("The required property '") + name + "' is missing");
    return valueParser(prop);
}

template <class ValueParser>
inline bool readOptionalProperty(
        Napi::Object o, const char *name, ValueParser valueParser,
        decltype (valueParser(Napi::Value()))& dst)
{
    auto prop = o.Get(name);
    if (prop.IsUndefined())
        return false;
    else {
        dst = valueParser(prop);
        return true;
    }
}

template <class T>
inline auto readRequiredProperty(Napi::Object o, const char *name)
{
    return readRequiredProperty(o, name, [](const Napi::Value& prop) {
        return toTInitializer<T>(prop.As<Ntype_t<T>>());
    });
}

template <class T>
inline bool readOptionalProperty(
        Napi::Object o, const char *name, T& dst)
{
    return readOptionalProperty(o, name, [](const Napi::Value& prop) {
        return toTInitializer<T>(prop.As<Ntype_t<T>>());
    }, dst);
}

template<unsigned int N, class T>
inline auto readRequiredFixedLengthArrayProperty(Napi::Object o, const char *name) {
    return readRequiredProperty(o, name, fixedLengthArrayObjReader<N, T>());
}

template<class T>
inline auto readRequiredArrayProperty(Napi::Object o, const char *name) {
    return readRequiredProperty(o, name, arrayObjReader<T>());
}



template<unsigned int N, class T>
struct Ntype<s3dmm::MultiIndex<N,T>>
{
    using type = Napi::Object;
    static Napi::Object toNvalue(Napi::Env env, const s3dmm::MultiIndex<N,T>& t) {
        return makeArrayObj(env, t);
    }
    static s3dmm::MultiIndex<N,T> toTInitializer(const Napi::Object& n)
    {
        return readFixedLengthArrayObj<N,T>(n);
    }
};



template<class T>
struct Ntype<std::vector<T>>
{
    using type = Napi::Object;

    static Napi::Object toNvalue(Napi::Env env, const std::vector<T>& t) {
        return makeArrayObj(env, t);
    }

    static std::vector<T> toTInitializer(const Napi::Object& n) {
        return readArrayObj<T>(n);
    }
};



template<class K, class V>
struct Ntype<std::map<K,V>>
{
    using type = Napi::Object;

    static Napi::Object toNvalue(Napi::Env env, const std::map<K,V>& t)
    {
        auto result = Napi::Object::New(env);
        for (auto& item : t) {
            auto keyStr = ::toNvalue(env, item.first).ToString();
            result[keyStr] = ::toNvalue(env, item.second);
        }
        return result;
    }

    static std::map<K,V> toTInitializer(const Napi::Object& n)
    {
        std::map<K,V> result;
        auto names = n.GetPropertyNames();
        auto length = names.Length();
        for (std::size_t i=0; i<length; ++i) {
            auto keyVal = names.Get(i);
            auto key = ::toTInitializer<K>(Ntype<K>::coerce(keyVal));
            result[key] = ::toTInitializer<V>(n.Get(keyVal).As<Ntype_t<V>>());
        }
        return result;
    }
};



template<class Tprop, class Thost, class L>
class PropClassAccessorHelper
{
public:
    using ThisClass = PropClassAccessorHelper<Tprop, Thost, L>;
    using SyncHost = silver_bullets::sync::SyncAccessor<Thost, L>;
    using value_type = typename Tprop::value_type;
    using N = Ntype_t<value_type>;

    static Napi::Value getter(const Napi::CallbackInfo& info)
    {
        try {
            auto shost = reinterpret_cast<SyncHost*>(info.Data());
            if (shost) {
                auto a = shost->access();
                Tprop& t = a.get();
                return toNvalue(info.Env(), t.get());
            }
            else
                throw std::runtime_error("no underlying input object has been set");
        }
        catch(const std::exception& e) {
            Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
            return Napi::Value();
        }
    }

    static void setter(const Napi::CallbackInfo& info)
    {
        try {
            auto shost = reinterpret_cast<SyncHost*>(info.Data());
            if (shost) {
                auto a = shost->access();
                Tprop& t = a.get();
                t.set(toTInitializer<value_type>(info[0].As<N>()));
            }
            else
                throw std::runtime_error("no underlying input object has been set");
        }
        catch(const std::exception& e) {
            Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
        }
    }

    static Napi::PropertyDescriptor accessor(
            const std::string& name,
            SyncHost& shost,
            napi_property_attributes attributes = napi_default)
    {
        return Napi::PropertyDescriptor::Accessor<&ThisClass::getter, &ThisClass::setter>(
                    name, attributes, &shost);
    }
};

template<class Tprop, class Thost, class L>
inline Napi::PropertyDescriptor makePropClassAccessor(
        const std::string& name,
        silver_bullets::sync::SyncAccessor<Thost, L>& shost,
        napi_property_attributes attributes = napi_enumerable)
{
    return PropClassAccessorHelper<Tprop, Thost, L>::accessor(name, shost, attributes);
}
