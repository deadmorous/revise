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

#include <iostream>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <utility>
#include <boost/noncopyable.hpp>
#include "MultiIndex.hpp"

namespace s3dmm {

// https://stackoverflow.com/questions/257288/is-it-possible-to-write-a-template-to-check-for-a-functions-existence

template<class>
struct sfinae_true : std::true_type{};

namespace detail {
  template<class T, class A0>
  static auto test_read(int)
      -> sfinae_true<decltype(std::declval<T>().read(std::declval<A0>()))>;
  template<class, class A0>
  static auto test_read(long) -> std::false_type;

  template<class T, class A0>
  static auto test_write(int)
      -> sfinae_true<decltype(std::declval<T>().write(std::declval<A0>()))>;
  template<class, class A0>
  static auto test_write(long) -> std::false_type;

} // detail::

template<class T, class Arg>
struct has_read : decltype(detail::test_read<T, Arg>(0)){};

template<class T, class Arg>
struct has_write : decltype(detail::test_write<T, Arg>(0)){};

template<class S> struct StreamExceptionSetter {
    static void setExceptions(S&) {}
};

template<> struct StreamExceptionSetter<std::ostream> {
    static void setExceptions(std::ostream& s) {
        s.exceptions(std::ostream::failbit);
    }
};

template<> struct StreamExceptionSetter<std::istream> {
    static void setExceptions(std::istream& s) {
        s.exceptions(std::istream::failbit);
    }
};

template<class S>
inline void setStreamExceptions(S& s) {
    StreamExceptionSetter<S>::setExceptions(s);
}

template <class Stream, class VectorSize, class StringSize>
class BinaryWriterTemplate : boost::noncopyable
{
public:
    using vector_size_type = VectorSize;
    using string_size_type = StringSize;
    explicit BinaryWriterTemplate(Stream& s) : m_s(s) {
        setStreamExceptions(m_s);
    }
    Stream& stream() {
        return m_s;
    }
private:
    Stream& m_s;
};

using BinaryWriter = BinaryWriterTemplate<std::ostream, std::size_t, std::size_t>;

template <class Stream, class VectorSize, class StringSize>
class BinaryReaderTemplate : boost::noncopyable
{
public:
    using vector_size_type = VectorSize;
    using string_size_type = StringSize;
    explicit BinaryReaderTemplate(Stream& s) : m_s(s) {
        setStreamExceptions(m_s);
    }
    Stream& stream() {
        return m_s;
    }
    template<class T>
    T read() {
        T result;
        *this >> result;
        return result;
    }
    bool readFloatAsDouble() const {
        return m_readFloatAsDouble;
    }
    void setReadFloatAsDouble(bool readFloatAsDouble) {
        m_readFloatAsDouble = readFloatAsDouble;
    }
private:
    Stream& m_s;
    bool m_readFloatAsDouble = false;
};

using BinaryReader = BinaryReaderTemplate<std::istream, std::size_t, std::size_t>;

namespace detail {

template<class S, class VS, class SS, class Container>
inline void writeContainer(BinaryWriterTemplate<S, VS, SS>& writer, const Container& c) {
    writer << static_cast<VS>(c.size());
    for (const auto& item : c) {
        writer << item;
    }
}

}

template <class S, class VS, class SS, class T>
inline std::enable_if_t<std::is_arithmetic<T>::value, BinaryWriterTemplate<S, VS, SS>&>
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const T& x) {
    writer.stream().write(reinterpret_cast<const char*>(&x), sizeof(T));
    return writer;
}

template <class S, class VS, class SS>
inline BinaryWriterTemplate<S, VS, SS>&
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const std::string& x) {
    writer << static_cast<SS>(x.size());
    writer.stream().write(x.data(), x.size());
    return writer;
}

template <class S, class VS, class SS, class T>
inline BinaryWriterTemplate<S, VS, SS>&
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const std::vector<T>& x) {
    detail::writeContainer(writer, x);
    return writer;
}

template <class S, class VS, class SS, class K, class V, class H>
inline BinaryWriterTemplate<S, VS, SS>&
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const std::unordered_map<K, V, H>& x) {
    detail::writeContainer(writer, x);
    return writer;
}

template <class S, class VS, class SS, class T1, class T2>
inline BinaryWriterTemplate<S, VS, SS>&
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const std::pair<T1, T2>& x) {
    writer << x.first << x.second;
    return writer;
}

template <class S, class VS, class SS, class T>
inline std::enable_if_t<has_write<T, BinaryWriterTemplate<S, VS, SS>&>::value, BinaryWriterTemplate<S, VS, SS>&>
operator<<(BinaryWriterTemplate<S, VS, SS>& writer, const T& x) {
    x.write(writer);
    return writer;
}

template <class S, class VS, class SS, class T>
inline std::enable_if_t<std::is_arithmetic<T>::value && !std::is_same<T, float>::value, BinaryReaderTemplate<S, VS, SS>&>
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, T& x) {
    reader.stream().read(reinterpret_cast<char*>(&x), sizeof(T));
    return reader;
}

template <class S, class VS, class SS>
inline BinaryReaderTemplate<S, VS, SS>&
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, float& x) {
    if (reader.readFloatAsDouble()) {
        double xd;
        reader.stream().read(reinterpret_cast<char*>(&xd), sizeof(double));
        x = static_cast<float>(xd);
    }
    else
        reader.stream().read(reinterpret_cast<char*>(&x), sizeof(float));
    return reader;
}

template <class S, class VS, class SS>
BinaryReaderTemplate<S, VS, SS>&
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, std::string& x) {
    SS size;
    reader >> size;
    std::vector<char> buf(size);
    reader.stream().read(buf.data(), size);
    x.assign(buf.begin(), buf.end());
    return reader;
}

template <class S, class VS, class SS, class T>
inline BinaryReaderTemplate<S, VS, SS>&
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, std::vector<T>& x) {
    x.resize(reader.template read<VS>());
    for (auto& item : x)
        reader >> item;
    return reader;
}

template <class S, class VS, class SS, class K, class V, class H>
inline BinaryReaderTemplate<S, VS, SS>&
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, std::unordered_map<K, V, H>& x) {
    x.clear();
    auto size = reader.template read<VS>();
    x.reserve(size);
    for (std::size_t i=0; i<size; ++i) {
        std::pair<K,V> item;
        reader >> item;
        x[item.first] = item.second;
    }

    return reader;
}

template <class S, class VS, class SS, class T1, class T2>
inline BinaryReaderTemplate<S, VS, SS>&
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, std::pair<T1, T2>& x) {
    reader >> const_cast<std::remove_const_t<T1>&>(x.first) >> x.second;
    return reader;
}

template <class S, class VS, class SS, class T>
inline std::enable_if_t<has_read<T, BinaryReaderTemplate<S, VS, SS>&>::value, BinaryReaderTemplate<S, VS, SS>&>
operator>>(BinaryReaderTemplate<S, VS, SS>& reader, T& x) {
    x.read(reader);
    return reader;
}

} // s3dmm
