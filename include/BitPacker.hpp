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

#include <boost/assert.hpp>
#include <type_traits>

namespace s3dmm {

template<unsigned int BitCount> struct BitPackContainer;
template<unsigned int BitCount> using BitPackContainer_t = typename BitPackContainer<BitCount>::type;

template<> struct BitPackContainer<2> { using type = unsigned char; };
template<> struct BitPackContainer<4> { using type = unsigned char; };
template<> struct BitPackContainer<8> { using type = unsigned char; };

template <unsigned int BitCount>
struct GenericBitCounter
{
    using value_type = BitPackContainer_t<BitCount>;
    static unsigned int countOnes(value_type value)
    {
        auto result = 0u;
        for (auto i=0u; i<BitCount; ++i, value>>=1)
            result += value & 1;
        return result;
    }

    static unsigned int countHighestOnes(value_type value, unsigned int highBitCount)
    {
        auto result = 0u;
        value_type mask = 1 << (BitCount-1);
        for (auto i=0u; i<highBitCount; ++i, mask>>=1)
            result += (value & mask) >> (BitCount-i-1);
        return result;
    }
};

namespace detail {

template <unsigned int C> inline constexpr unsigned int multiplyBy(unsigned int x) { return C * x; }
template <> inline constexpr unsigned int multiplyBy<1>(unsigned int x) { return x; }
template <> inline constexpr unsigned int multiplyBy<2>(unsigned int x) { return x << 1; }
template <> inline constexpr unsigned int multiplyBy<4>(unsigned int x) { return x << 2; }
template <> inline constexpr unsigned int multiplyBy<8>(unsigned int x) { return x << 3; }

inline constexpr unsigned int log2(unsigned int x)
{
    BOOST_ASSERT(x != 0);
    auto result = 0u;
    for (; x!=1u; x>>=1) {
        BOOST_ASSERT((x & 1) == 0);
        ++result;
    }
    return result;
}

} // detail

template <unsigned int BitCount>
class CachedBitCounter
{
public:
    static constexpr const unsigned int bit_count = BitCount;

    using value_type = BitPackContainer_t<BitCount>;
    static unsigned int countOnes(value_type value) {
        auto& sd = staticData();
        BOOST_ASSERT(sd.initialized);
        return sd.ones[value];
    }

    static unsigned int countHighestOnes(value_type value, unsigned int highBitCount)
    {
        auto& sd = staticData();
        BOOST_ASSERT(sd.initialized);
        auto uvalue = static_cast<unsigned int>(value);
        return sd.highestOnes[detail::multiplyBy<BitCount>(uvalue) + uvalue + highBitCount];
    }

    static void init()
    {
        auto& sd = staticData();
        BOOST_ASSERT(!sd.initialized);
        value_type value = 0;
        auto iho = 0u;
        for (auto io=0u; io<ValueCount; ++io, ++value) {
            sd.ones[io] = GenericBitCounter<BitCount>::countOnes(value);
            for (auto ih=0u; ih<=BitCount; ++ih, ++iho)
                sd.highestOnes[iho] = GenericBitCounter<BitCount>::countHighestOnes(value, ih);
        }
        sd.initialized = true;
    }

    static void maybeInit()
    {
        if (!staticData().initialized)
            init();
    }

private:
    static constexpr const unsigned int ValueCount = 1u << BitCount;
    struct StaticData {
        unsigned int ones[ValueCount];
        unsigned int highestOnes[ValueCount*(BitCount+1u)];
        bool initialized;
    };
    static StaticData& staticData() {
        static StaticData staticData;
        return staticData;
    }
};

template<unsigned int BitCount, class T,
         std::enable_if_t<std::is_same<std::remove_cv_t<T>, BitPackContainer_t<BitCount>>::value, int> = 0>
class AlignedBitPacker
{
public:
    using value_type = T;
    explicit AlignedBitPacker(value_type *data) :
        m_data(data)
    {}

    value_type get(std::size_t index) const {
        auto shift = (index & AddressShiftMask) << L2BitCount;
        return (m_data[index >> AddressShift] >> shift) & MaxValue;
    }

    void set(std::size_t index, value_type x) const
    {
        BOOST_ASSERT(x <= MaxValue);
        auto& value = m_data[index >> AddressShift];
        auto shift = (index & AddressShiftMask) << L2BitCount;
        auto mask = MaxValue << shift;
        value = (value & ~mask) | (x << shift);
    }

private:
    static constexpr const unsigned int MaxValue = (1 << BitCount) - 1;
    static constexpr const unsigned int ValuesPerContainer = (sizeof (value_type) << 3) / BitCount;
    static constexpr const unsigned int AddressShift = detail::log2(ValuesPerContainer);
    static constexpr const unsigned int AddressShiftMask = (1 << AddressShift) - 1;
    static constexpr const unsigned int L2BitCount = detail::log2(BitCount);
    value_type *m_data;
};

template<unsigned int BitCount, class T,
        std::enable_if_t<std::is_same<std::remove_cv_t<T>, BitPackContainer_t<BitCount>>::value, int> = 0>
class FullValueBitPacker
{
public:
    using value_type = T;
    explicit FullValueBitPacker(value_type *data) :
        m_data(data)
    {}

    value_type get(std::size_t index) const {
        return m_data[index];
    }

    void set(std::size_t index, value_type x) const {
        m_data[index] = x;
    }

private:
    value_type *m_data;
};

template <unsigned int BitCount, class T> struct BitPacker;
template <unsigned int BitCount, class T> using BitPacker_t = typename BitPacker<BitCount, T>::type;

template <unsigned int BitCount> using ConstBitPacker = BitPacker<BitCount, const BitPackContainer_t<BitCount>>;
template <unsigned int BitCount> using ConstBitPacker_t = typename ConstBitPacker<BitCount>::type;
template <unsigned int BitCount> using MutableBitPacker = BitPacker<BitCount, BitPackContainer_t<BitCount>>;
template <unsigned int BitCount> using MutableBitPacker_t = typename MutableBitPacker<BitCount>::type;

template<class T> struct BitPacker<2, T> { using type = AlignedBitPacker<2, T>; };
template<class T> struct BitPacker<4, T> { using type = AlignedBitPacker<4, T>; };
template<class T> struct BitPacker<8, T> { using type = FullValueBitPacker<8, T>; };

template<unsigned int BitCount, class T>
inline BitPacker_t<BitCount, T> makeBitPacker(T *data) {
    return BitPacker_t<BitCount, T>(data);
}

} // s3dmm
