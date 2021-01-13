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

#include "./RecentItemCache.hpp"

#include <mutex>

namespace s3dmm {

template<class K, class V, class Lockable>
class SyncRecentItemCache
{
public:
    using key_type = K;
    using value_type = V;
    explicit SyncRecentItemCache(Lockable& lockable) :
      m_lockable(lockable)
    {}

    SyncRecentItemCache(Lockable& lockable, unsigned int maxSize) :
      m_cache(maxSize),
      m_lockable(lockable)
    {}

    template<class F>
    V& get(const K& key, F generator)
    {
        std::unique_lock<Lockable> lock(m_lockable);
        auto maybe = m_cache.maybeGet(key);
        if (maybe)
            return *maybe;
        else {
            lock.unlock();
            return get(key, generator());
        }
    }

private:
    Lockable& m_lockable;
    RecentItemCache<K, V> m_cache;

    V& get(const K& key, V&& value)
    {
        std::unique_lock<Lockable> lock(m_lockable);
        auto maybe = m_cache.maybeGet(key);
        if (maybe)
            return *maybe;
        else
            return m_cache.emplace(key, std::move(value));
    }
};

} // s3dmm
