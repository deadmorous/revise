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

#include "VsControllerFrameOutput.hpp"
#include "VsControllerFrameOutputHeader.hpp"

namespace s3vs
{

class VsControllerFrameOutputRW
{
public:
    VsControllerFrameOutputRW(SharedMemory* shmem, unsigned int frameSize) :
        m_shmem(shmem), m_frameSize(frameSize)
    {}
    explicit VsControllerFrameOutputRW(
        const VsControllerFrameOutput& frameOutput) :
        VsControllerFrameOutputRW(frameOutput.shmem.get(), frameOutput.frameSize)
    {
        m_shmemPtr = frameOutput.shmem;
    }

    bool readFrame(VsControllerFrameOutputHeader& hdr, void* image = nullptr) const
    {
        return readFrame(hdr, image, [](const auto&) { return true; });
    }

    template<class F>
    bool readFrame(VsControllerFrameOutputHeader& hdr, void* image, const F& needImage) const
    {
        BOOST_ASSERT(m_shmem);
        auto acc = m_shmem->readAccessor();
        BOOST_ASSERT(sizeof(hdr) + m_frameSize <= m_shmem->size());
        memcpy(&hdr, acc.get(), sizeof(hdr));
        if (image && m_frameSize && needImage(hdr)) {
            memcpy(image, reinterpret_cast<const unsigned char*>(acc.get()) + sizeof(hdr),
                   m_frameSize);
            return true;
        }
        else
            return false;
    }

    bool writeFrame(const VsControllerFrameOutputHeader& hdr, const void* image = nullptr)
    {
        BOOST_ASSERT(m_shmem);
        auto acc = m_shmem->writeAccessor();
        BOOST_ASSERT(sizeof(hdr) + m_frameSize <= m_shmem->size());
        memcpy(acc.get(), &hdr, sizeof(hdr));
        if (image && m_frameSize) {
            memcpy(reinterpret_cast<unsigned char*>(acc.get()) + sizeof(hdr), image,
                   m_frameSize);
            return true;
        }
        else
            return false;
    }

private:
    SharedMemory* m_shmem{nullptr};
    std::shared_ptr<SharedMemory> m_shmemPtr;
    unsigned int m_frameSize{0};
};

} // namespace s3vs
