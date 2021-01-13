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

#include "MultiIndex.hpp"

#include <boost/iterator/iterator_facade.hpp>

namespace s3dmm {

template<unsigned int N, class Func>
class ImageStencil
{
public:
    using Index = MultiIndex<N, unsigned int>;
    class iterator : public boost::iterator_facade<
                         iterator, const Index, boost::forward_traversal_tag>
    {
    public:
        iterator(const Index& size, const Func& func, const Index& index) :
            m_size(size),
            m_func(func),
            m_index(index)
        {}

        const Index& dereference() const {
            return m_index;
        }

        void increment() {
            if (incMultiIndex(m_index, Index(), m_size))
                skipVoid();
            else
                m_index = m_size;
        }

        bool equal(const iterator& that) const
        {
            BOOST_ASSERT(that.m_size == m_size);
            return m_index == that.m_index;
        }

    private:
        Index m_size;
        Func m_func;
        Index m_index;

        void skipVoid() {
            while (!m_func(m_index)) {
                if (!incMultiIndex(m_index, Index(), m_size)) {
                    m_index = m_size;
                    return;
                }
            }
        }
    };
    using const_iterator = iterator;

    ImageStencil(const Index& size, const Func& func) :
        m_size(size),
        m_func(func)
    {}

    iterator begin() const
    {
        iterator result(m_size, m_func, Index());
        if (!m_func(*result))
            ++result;
        return result;
    }

    iterator end() const {
        return iterator(m_size, m_func, m_size);
    }

    Index size() const {
        return m_size;
    }

private:
    Index m_size;
    Func m_func;
};

} // s3dmm
