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

#include "binary_io.hpp"
#include "real_type.hpp"
#include "binary_io.hpp"

struct BinaryMeshWriterHelper
{
public:
    BinaryMeshWriterHelper(std::ostream& s, unsigned int zoneCount) :
        m_bw(s),
        m_zoneCount(zoneCount),
        m_nextZoneIndex(0)
    {}
    void writeFileHeader (const std::vector<std::string>& fields)
    {
        using namespace std;
        m_bw << string("s3dmm_binary_79fbed93-3277-4ee4-accb-6f156987aa7a"); // signature
        m_bw << fields << m_zoneCount;
        m_zoneDataPos = m_bw.stream().tellp();
        for (size_t i=0; i<m_zoneCount; ++i)
            m_bw << streamoff(0);
    }
    void writeZoneHeader (unsigned int i)
    {
        writeZonePos();
        m_bw << size_t(1) << size_t(i);
    }
    void writeZoneHeader (unsigned int i, unsigned int j)
    {
        writeZonePos();
        m_bw << size_t(2) << size_t(i) << size_t(j);
    }
    void writeZoneHeader (unsigned int i, unsigned int j, unsigned int k)
    {
        writeZonePos();
        m_bw << size_t(3) << size_t(i) << size_t(j) << size_t(k);
    }
    void writeZoneHeader (const std::vector<unsigned int>& ijk)
    {
        writeZonePos();
        BOOST_ASSERT(ijk.size() >= 1 && ijk.size() <= 3);
        m_bw << ijk.size();
        for (auto i : ijk)
            m_bw << size_t(i);
    }

    void writeRow(s3dmm::real_type firstArg) {
        m_bw << firstArg;
    }
    template<class ... RestArgs>
    void writeRow(s3dmm::real_type firstArg, s3dmm::real_type secondArg, RestArgs ... restArgs) {
        m_bw << firstArg;
        writeRow(secondArg, restArgs ...);
    }
    template<class It,
             class = std::enable_if_t<
                std::is_same<
                     std::remove_cv_t< std::remove_reference_t<decltype (*It())>>,
                     s3dmm::real_type>::value, int>>
    void writeRow(It begin, It end) {
        for (; begin!=end; ++begin)
            m_bw << *begin;
    }

private:
    s3dmm::BinaryWriter m_bw;
    size_t m_zoneCount;
    size_t m_nextZoneIndex;
    std::streamoff m_zoneDataPos;

    void writeZonePos()
    {
        BOOST_ASSERT(m_nextZoneIndex < m_zoneCount);
        auto& s = m_bw.stream();
        auto pos = s.tellp();
        s.seekp(m_zoneDataPos + m_nextZoneIndex*sizeof(size_t));
        m_bw << static_cast<size_t>(pos);
        s.seekp(pos);
        ++m_nextZoneIndex;
    }
};

