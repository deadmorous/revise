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

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>

#include <atomic>
#include <cstring>
#include <random>
#include <stdexcept>
#include <iostream>
#include <memory>

#include <boost/noncopyable.hpp>
#include <boost/assert.hpp>

class SharedMemory : boost::noncopyable
{

public:
    SharedMemory() = default;
    SharedMemory(SharedMemory&& that) :
        m_shmid(that.m_shmid),
        m_ptr(that.m_ptr),
        m_autoDelete(that.m_autoDelete)
    {
        that.m_shmid = InvalidShmid;
        that.m_autoDelete = false;
    }

    SharedMemory& operator=(SharedMemory&& that)
    {
        m_shmid = that.m_shmid;
        m_ptr = that.m_ptr;
        m_autoDelete = that.m_autoDelete;
        that.m_shmid = InvalidShmid;
        return *this;
    }

    static SharedMemory attach(int shmid) {
        return SharedMemory(shmid);
    }

    static std::shared_ptr<SharedMemory> attachNew(int shmid) {
        return std::shared_ptr<SharedMemory>(new SharedMemory(shmid));
    }

    static SharedMemory create(std::size_t byteCount) {
        return SharedMemory(byteCount);
    }

    static std::shared_ptr<SharedMemory> createNew(std::size_t byteCount) {
        return std::shared_ptr<SharedMemory>(new SharedMemory(byteCount));
    }

    ~SharedMemory()
    {
        auto shmid = m_shmid;
        auto autoDelete = m_autoDelete;
        detach();
        // Remove shmem if autoDelete is true
        if (autoDelete && shmctl(shmid, IPC_RMID, nullptr) != 0) {
            // TODO: Handle error somehow
            // throw std::runtime_error("SharedMemory::~SharedMemory(): shmctl() failed");
        }
    }

    void detach()
    {
        if (m_shmid != InvalidShmid) {
            BOOST_ASSERT(isValidShmatResult(m_ptr));
            if (shmdt(m_ptr) != 0)
                throw std::runtime_error("SharedMemory::detach(): shmdt() failed");
            m_shmid = InvalidShmid;
            m_autoDelete = false;
        }
    }

    operator bool() const noexcept {
        return m_shmid != InvalidShmid;
    }

    int shmid() const noexcept {
        return m_shmid;
    }

    std::size_t size() const noexcept {
        return m_shmid == InvalidShmid? 0: sizeRef();
    }

    void read(void *dst, std::size_t start, std::size_t count) const noexcept
    {
        BOOST_ASSERT(m_shmid != InvalidShmid);
        ReadLock lock(controlBlock());
        memcpy(dst, userMemory() + start, count);
    }

    void write(const void *src, std::size_t start, std::size_t count) noexcept
    {
        BOOST_ASSERT(m_shmid != InvalidShmid);
        WriteLock lock(controlBlock());
        memcpy(userMemory() + start, src, count);
    }

    void read(void *dst) const noexcept {
        read(dst, 0, size());
    }

    void write(const void *src) noexcept {
        write(src, 0, size());
    }

private:
    class ReadLock : boost::noncopyable
    {
    public:
        explicit ReadLock(std::atomic<int>& ctlBlock) : m_ctlBlock(ctlBlock)
        {
            while (++m_ctlBlock < 0)
                --m_ctlBlock;

        }
        ~ReadLock() {
            --m_ctlBlock;
        }
    private:
        std::atomic<int>& m_ctlBlock;
    };

    class WriteLock: boost::noncopyable
    {
    public:
        WriteLock(std::atomic<int>& ctlBlock) : m_ctlBlock(ctlBlock)
        {
            int free = 0;
            int write = -MaxClientLimit;
            while (!m_ctlBlock.compare_exchange_weak(free, write)) {}
        }

        ~WriteLock() {
            m_ctlBlock += MaxClientLimit;
        }

    private:
        std::atomic<int>& m_ctlBlock;
    };

public:
    class ReadAccessor : boost::noncopyable
    {
    public:
        ReadAccessor(std::atomic<int>& ctlBlock, const void *data) :
            m_lock(ctlBlock), m_data(data)
        {}
        const void *get() const noexcept {
            return m_data;
        }
        operator const void*() const noexcept {
            return m_data;
        }
    private:
        ReadLock m_lock;
        const void *m_data;
    };

    class WriteAccessor : boost::noncopyable
    {
    public:
        WriteAccessor(std::atomic<int>& ctlBlock, void *data) :
            m_lock(ctlBlock), m_data(data)
        {}
        void *get() const noexcept {
            return m_data;
        }
        operator void*() const noexcept {
            return m_data;
        }
    private:
        WriteLock m_lock;
        void *m_data;
    };

    ReadAccessor readAccessor() const {
        return ReadAccessor(controlBlock(), userMemory());
    }

    WriteAccessor writeAccessor() {
        return WriteAccessor(controlBlock(), userMemory());
    }

private:
    static constexpr const int InvalidShmid = -1;
    static constexpr const int MaxClientLimit = 1000;
    static constexpr const std::ptrdiff_t ControlBlockOffset = sizeof(std::size_t);
    static constexpr const std::ptrdiff_t UserDataOffset = ControlBlockOffset + sizeof(std::atomic<int>);
    static constexpr const std::size_t ExtraByteCount = UserDataOffset;

    int m_shmid = InvalidShmid;
    void* m_ptr = nullptr;
    bool m_autoDelete = false;

    // Attaches
    explicit SharedMemory(int shmid) :
        m_shmid(shmid),
        m_ptr(shmid >= 0? shmat(shmid, nullptr, 0): nullptr),
        m_autoDelete(false)
    {
        if (shmid >= 0 && !isValidShmatResult(m_ptr))
            throw std::invalid_argument("Wrong Shared Memory Id passed, please check");
    }

    // Creates
    explicit SharedMemory(std::size_t byteCount) :
        m_shmid(allocShmem(byteCount)),
        m_ptr(shmat(m_shmid, nullptr, 0)),
        m_autoDelete(true)
    {
        BOOST_ASSERT(isValidShmatResult(m_ptr));
        sizeRef() = byteCount;
        controlBlock().store(0);
    }

    static bool isValidShmatResult(void *ptr) {
        return ptr != reinterpret_cast<void*>(std::size_t(-1));
    }

    static int allocShmem(std::size_t byteCount)
    {
        std::random_device rd;     // only used once to initialise (seed) engine
        std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
        std::uniform_int_distribution<int> uni(5000, 10000); // guaranteed unbiased
        /* Create the segment */
        int shmid = -1;
        while(shmid < 0)
        {
            int key_ = uni(rng);
            shmid = shmget(key_, ExtraByteCount + byteCount,
                           IPC_CREAT |  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
            if(shmid < 0 && errno != EEXIST)
                throw std::invalid_argument("Error creating shared memory segment");
        }
        return shmid;
    }

    std::size_t& sizeRef() const {
        return *reinterpret_cast<std::size_t*>(m_ptr);
    }

    std::atomic<int>& controlBlock() const {
        return *reinterpret_cast<std::atomic<int>*>(memory() + ControlBlockOffset);
    }

    char *userMemory() const {
        return memory() + UserDataOffset;
    }

    char *memory() const {
        return reinterpret_cast<char*>(m_ptr);
    }
};
