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

#include <vlCore/IMutex.hpp>

#include <mutex>

namespace s3vs
{

class VsWorkerInit
{
public:
    using mutex_type = std::recursive_mutex;

    VsWorkerInit() = default;
    VsWorkerInit(
        const std::string& shaderPath,
        bool logInfo,
        const std::string& logFileName)
    {
        initialize(shaderPath, logInfo, logFileName);
    }
    void initialize(
        const std::string& shaderPath,
        bool logInfo,
        const std::string& logFileName);

    static mutex_type& getMutex()
    {
        static mutex_type mut;
        return mut;
    }

    class Mutex : public vl::IMutex
    {
    public:
        Mutex(mutex_type& mut) : m_mut(mut) {}
        void lock() override
        {
            m_mut.lock();
        }
        void unlock() override
        {
            m_mut.unlock();
        }
        int isLocked() const override
        {
            return -1;
        }
    private:
        mutex_type& m_mut;
    };

    static Mutex& getIMutex()
    {
        static Mutex mut(getMutex());
        return mut;
    }
};


} // namespace s3vs
