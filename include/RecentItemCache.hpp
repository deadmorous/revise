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

#include <map>

namespace s3dmm {

template<class K, class V>
class RecentItemCache
{
public:
    using key_type = K;
    using value_type = V;
    static const unsigned int DefaultMaxSize = 1000;

    RecentItemCache() = default;
    explicit RecentItemCache(unsigned int maxSize) : m_maxSize(maxSize) {}

    template<class F>
    V& get(const K& key, F generator)
    {
        auto maybe = maybeGet(key);
        if (maybe)
            return *maybe;
        else
            return emplace(generator());
    }

    V *maybeGet(const K& key)
    {
        auto it = m_cache.find(key);
        if (it == m_cache.end())
            return nullptr;
        else
            return &getExisting(it);
    }

    V& emplace(const K& key, V&& value)
    {
        BOOST_ASSERT(m_cache.find(key) == m_cache.end());
        if (m_cache.size() == m_maxSize) {
            auto itts = m_ts2it.begin();
            m_cache.erase(itts->second);
            m_ts2it.erase(itts);
        }
        auto ts = m_newTimestamp++;
        auto it = m_ts2it[ts] = m_cache.emplace(
            typename CacheItemMap::value_type(key, CacheItemData(std::move(value), ts)))
                .first;
        return it->second.first;
    }

private:
    // first=original value, second=timestamp
    using CacheItemData = std::pair<V, unsigned int>;
    using CacheItemMap = std::map<K, CacheItemData>;
    unsigned int m_maxSize = DefaultMaxSize;
    unsigned int m_newTimestamp = 0;

    // key=original key, value=original value and timestamp
    CacheItemMap m_cache;

    // key=timestamp, value=m_cache iterator
    std::map<unsigned int, typename CacheItemMap::iterator> m_ts2it;

    V& getExisting(const typename CacheItemMap::iterator& it)
    {
        auto& ts = it->second.second;
        m_ts2it.erase(m_ts2it.find(ts));
        ts = m_newTimestamp++;
        m_ts2it[ts] = it;
        return it->second.first;
    }
};

} // s3dmm
