/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021-2024 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

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

#include "BackToFrontOrder.hpp"
#include "BoundingBox.hpp"
#include "UniformIndexRangeSplitter.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <map>

namespace
{

using namespace s3dmm;

template <unsigned int N>
using BBox = BoundingBox<N, real_type>;

// In the tests below, IndexedBlocksType is used as a template parameter
// to BackToFrontOrder. Therefore, it satisfies the IndexedBlocksType concept.
template <unsigned int N>
class IndexCube
{
public:

    using IndexVecTraits = ScalarOrMultiIndex<N, unsigned int>;
    using Block = IndexVecTraits::type;
    using IndexVec = IndexVecTraits::type;
    using Splitter = UniformIndexRangeSplitter;

    static constexpr auto dim = N;

    explicit IndexCube(unsigned int size,
                       const BBox<N>& bbox):
        m_size{ size },
        m_bbox{ bbox }
    {}

    Block at(const IndexVec& pos) const noexcept
    { return pos; }

    IndexVec begin_index() const noexcept
    {
        return IndexVecTraits::fromMultiIndex(
            MultiIndex<N, unsigned int>::filled(0) );
    }

    IndexVec end_index() const noexcept
    {
        return IndexVecTraits::fromMultiIndex(
            MultiIndex<N, unsigned int>::filled(m_size) );
    }

    Splitter index_range_splitter(unsigned int axis) const noexcept
    {
        BOOST_ASSERT(axis < dim);
        return { {0, m_size}, CoordRange::from_vec(m_bbox.range(axis)) };
    }

private:
    unsigned int m_size;
    BBox<N> m_bbox;
};


// BlockOrderPrinter prints the results of block ordering obtained using
// BackToFrontOrder. We use it to formulate assertions in the tests.
template <unsigned int N>
struct BlockOrderPrinter;


template <unsigned int N>
std::ostream& operator<<(std::ostream& s, const BlockOrderPrinter<N>& p)
{
    p.print(s);
    return s;
}


template <>
class BlockOrderPrinter<1>
{
public:
    BlockOrderPrinter(unsigned int eye_idx,
                      std::string_view decorators):
        m_eye_idx{ eye_idx },
        m_decorators{ decorators }
    {}

    explicit BlockOrderPrinter(unsigned int eye_idx):
        BlockOrderPrinter{ eye_idx, m_default_decorators }
    {}

    explicit BlockOrderPrinter():
        BlockOrderPrinter{ ~0u }
    {}

    void operator()(unsigned int pos, unsigned int ord)
    { m_pos2ord[pos] = ord; }

    void print(std::ostream& s) const
    {
        if (m_pos2ord.empty())
            return;

        auto pos = ~0u;

        auto print_pos = [&](unsigned int pos, unsigned int ord)
        {
            auto dec = m_decorators[pos == m_eye_idx ? 1 : 0];
            s << ' ' << dec << ' ';
            if (ord == ~0u)
                s << "---";
            else
                s << std::setw(3) << ord;
            s << ' ' << dec << ' ';
        };

        for (auto it=m_pos2ord.begin(), end=m_pos2ord.end(); it!=end; ++it)
        {
            if (pos == ~0u)
                pos = it->first;
            else
                for (++pos; pos!=it->first; ++pos)
                    print_pos(pos, ~0u);
            print_pos(pos, it->second);
        }
    }

private:
    static constexpr std::string_view m_default_decorators = " *";

    unsigned int m_eye_idx;
    std::string_view m_decorators;
    std::map<unsigned int, unsigned int> m_pos2ord;
};



template <>
class BlockOrderPrinter<2>
{
public:
    BlockOrderPrinter(const Vec2u& eye_idx,
                      std::string_view decorators):
        m_eye_idx{ eye_idx },
        m_decorators{ decorators }
    {}

    BlockOrderPrinter(const Vec2u& eye_idx):
        BlockOrderPrinter{ eye_idx, m_default_decorators }
    {}

    explicit BlockOrderPrinter():
        BlockOrderPrinter{ { ~0u, ~0u } }
    {}

    void operator()(const Vec2u& pos, unsigned int ord)
    {
        auto col = pos[0];
        auto row = pos[1];

        if (!m_rows.contains(row))
        {
            auto dec = m_decorators.substr(row == m_eye_idx[1] ? 1 : 0);
            m_rows[row] = BlockOrderPrinter<1>{ m_eye_idx[0], dec };
        }
        m_rows[row](col, ord);
    }

    void print(std::ostream& s) const
    {
        if (m_rows.empty())
            return;

        auto row = ~0u;

        for (auto it=m_rows.begin(), end=m_rows.end(); it!=end; ++it)
        {
            if (row == ~0u)
                row = it->first;
            else
                for (++row; row!=it->first; ++row)
                    s << std::endl;
            s << it->second << std::endl;
        }
    }

private:
    static constexpr std::string_view m_default_decorators = " *#";

    Vec2u m_eye_idx;
    std::string_view m_decorators;
    std::map<unsigned int, BlockOrderPrinter<1>> m_rows;
};



template <>
class BlockOrderPrinter<3>
{
public:
    BlockOrderPrinter(const Vec3u& eye_idx,
                      std::string_view decorators):
        m_eye_idx{ eye_idx },
        m_decorators{ decorators }
    {}

    BlockOrderPrinter(const Vec3u& eye_idx):
        BlockOrderPrinter{ eye_idx, m_default_decorators }
    {}

    explicit BlockOrderPrinter():
        BlockOrderPrinter{ { ~0u, ~0u, ~0u } }
    {}

    void operator()(const Vec3u& pos, unsigned int ord)
    {
        auto col = pos[0];
        auto row = pos[1];
        auto plane = pos[2];

        if (!m_planes.contains(plane))
        {
            auto dec = m_decorators.substr(plane == m_eye_idx[2] ? 1 : 0);
            m_planes[plane] = BlockOrderPrinter<2>{
                { m_eye_idx[0], m_eye_idx[1] }, dec };
        }
        m_planes[plane]({col, row}, ord);
    }

    void print(std::ostream& s) const
    {
        for (auto it=m_planes.begin(), end=m_planes.end(); it!=end; ++it)
        {
            s << "Plane " << it->first << ":\n";
            s << it->second << std::endl;
        }
    }

private:
    static constexpr std::string_view m_default_decorators = " *#@";

    Vec3u m_eye_idx;
    std::string_view m_decorators;
    std::map<unsigned int, BlockOrderPrinter<2>> m_planes;
};


// format_b2fo_result formats the result of ordering of the blocks of
// IndexCube<N> with specified size and bounding box with all edges [0, 1];
// the eye parameter is passed to BackToFrontOrder::range.
template <unsigned int N>
    std::string format_b2fo_result(
        unsigned int size,
        const ScalarOrMultiIndex_t<N, real_type>& eye )
{
    using RVT = ScalarOrMultiIndex<N, real_type>;
    using RV = MultiIndex<N, real_type>;
    using IVT = ScalarOrMultiIndex<N, unsigned int>;
    using IV = MultiIndex<N, unsigned int>;

    // Define index cube
    auto bbox =
        BBox<N>{}
                << RVT::fromMultiIndex(RV::filled(0))
                << RVT::fromMultiIndex(RV::filled(1));
    auto icube = IndexCube{ size, bbox };

    // Initialize block order printer
    auto eye_idx = typename IVT::type{};
    for (unsigned int axis=0; axis<N; ++axis)
        IVT::element(eye_idx, axis) =
            icube.index_range_splitter(axis)(RVT::element(eye, axis)).at;
    auto p = BlockOrderPrinter<N>{ eye_idx };

    // Record the result of block ordering
    BackToFrontOrder b2fo{ icube };
    size_t i = 0;
    for (auto block : b2fo.range(eye))
        p(block, i++);

    // Format the result of block ordering
    std::ostringstream s;
    s << p;
    return s.str();
};


// formats test assertion. Arguments are same as for format_b2fo_result.
// The outut of this function should be checked manually and copy-pasted
// to a test function.
template <unsigned int N>
std::string ft(
    unsigned int size,
    const ScalarOrMultiIndex_t<N, real_type>& eye )
{
    auto str = format_b2fo_result<N>(size, eye);
    std::ostringstream result;
    auto v = [&](const ScalarOrMultiIndex_t<N, real_type>& x)
    {
        std::ostringstream s;
        if constexpr (N==1)
            s << x;
        else
        {
            s << '{';
            for (unsigned int axis=0; axis<N; ++axis)
            {
                if (axis > 0)
                    s << ',';
                s << ' ';
                s << x[axis];
            }
            s << " }";
        }
        return s.str();
    };
    result
        << "    EXPECT_EQ(\n"
        << "        format_b2fo_result<" << N << ">("
        << size << ", " << v(eye) << "),\n";

    [[maybe_unused]] auto LLL = str.size();

    for (size_t pos=0; pos!=std::string::npos;)
    {
        auto pos2 = str.find_first_of('\n', pos);
        auto line = str.substr(pos, pos2 == std::string::npos ?pos2: pos2-pos);
        if (pos2 == std::string::npos && line.empty())
        {
            result << ");\n";
            break;
        }

        if (pos > 0)
            result << '\n';
        result << "        \"" << line;
        if (pos2 != std::string::npos)
            result << "\\n";
        result << "\"";
        if (pos2 == std::string::npos)
            pos = str.size();
        else
            pos = pos2 + 1;
    }

    return result.str();
}

// Tests BackToFrontOrder<1>.
TEST(BackToFrontOrderTest, Basic_1d)
{
    /*
    Assertions below are generated using the following code,
    and then tested manually.

    for (auto x : {-1., 0.01, 0.49, 0.99, 2.})
        std::cout << ft<1>(7, x) << std::endl;
    */

    EXPECT_EQ(
        format_b2fo_result<1>(7, -1),
        "     6        5        4        3        2        1        0   ");

    EXPECT_EQ(
        format_b2fo_result<1>(7, 0.01),
        " *   6 *      5        4        3        2        1        0   ");

    EXPECT_EQ(
        format_b2fo_result<1>(7, 0.49),
        "     0        1        2    *   6 *      5        4        3   ");

    EXPECT_EQ(
        format_b2fo_result<1>(7, 0.99),
        "     0        1        2        3        4        5    *   6 * ");

    EXPECT_EQ(
        format_b2fo_result<1>(7, 2),
        "     0        1        2        3        4        5        6   ");
}

// Tests BackToFrontOrder<2>.
TEST(BackToFrontOrderTest, Basic_2d)
{
    /*
    Assertions below are generated using the following code,
    and then tested manually.

    real_type coord[] = { -1, 0.1, 0.5, 0.9, 2 };
    for (auto y : coord)
        for (auto x : coord)
            std::cout << ft<2>(5, { x,  y }) << std::endl;
    */


    EXPECT_EQ(
        format_b2fo_result<2>(5, { -1, -1 }),
        "    24       23       21       18       14   \n"
        "    22       20       17       13        9   \n"
        "    19       16       12        8        5   \n"
        "    15       11        7        4        2   \n"
        "    10        6        3        1        0   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.1, -1 }),
        " *  24 *     19       18       16       13   \n"
        " *  23 *     17       15       12        9   \n"
        " *  22 *     14       11        8        5   \n"
        " *  21 *     10        7        4        2   \n"
        " *  20 *      6        3        1        0   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.5, -1 }),
        "     8        9    *  24 *     19       18   \n"
        "     6        7    *  23 *     17       16   \n"
        "     4        5    *  22 *     15       14   \n"
        "     2        3    *  21 *     13       12   \n"
        "     0        1    *  20 *     11       10   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.9, -1 }),
        "    13       16       18       19    *  24 * \n"
        "     9       12       15       17    *  23 * \n"
        "     5        8       11       14    *  22 * \n"
        "     2        4        7       10    *  21 * \n"
        "     0        1        3        6    *  20 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 2, -1 }),
        "    14       18       21       23       24   \n"
        "     9       13       17       20       22   \n"
        "     5        8       12       16       19   \n"
        "     2        4        7       11       15   \n"
        "     0        1        3        6       10   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { -1, 0.1 }),
        " *  24 *  *  23 *  *  22 *  *  21 *  *  20 * \n"
        "    19       18       16       13        9   \n"
        "    17       15       12        8        5   \n"
        "    14       11        7        4        2   \n"
        "    10        6        3        1        0   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.1, 0.1 }),
        " #  24 #  *  19 *  *  18 *  *  17 *  *  16 * \n"
        " *  23 *     15       14       12        9   \n"
        " *  22 *     13       11        8        5   \n"
        " *  21 *     10        7        4        2   \n"
        " *  20 *      6        3        1        0   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.5, 0.1 }),
        " *  16 *  *  17 *  #  24 #  *  19 *  *  18 * \n"
        "     6        7    *  23 *     15       14   \n"
        "     4        5    *  22 *     13       12   \n"
        "     2        3    *  21 *     11       10   \n"
        "     0        1    *  20 *      9        8   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.9, 0.1 }),
        " *  16 *  *  17 *  *  18 *  *  19 *  #  24 # \n"
        "     9       12       14       15    *  23 * \n"
        "     5        8       11       13    *  22 * \n"
        "     2        4        7       10    *  21 * \n"
        "     0        1        3        6    *  20 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 2, 0.1 }),
        " *  20 *  *  21 *  *  22 *  *  23 *  *  24 * \n"
        "     9       13       16       18       19   \n"
        "     5        8       12       15       17   \n"
        "     2        4        7       11       14   \n"
        "     0        1        3        6       10   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { -1, 0.5 }),
        "     7        5        3        1        0   \n"
        "     9        8        6        4        2   \n"
        " *  24 *  *  23 *  *  22 *  *  21 *  *  20 * \n"
        "    19       18       16       14       12   \n"
        "    17       15       13       11       10   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.1, 0.5 }),
        " *  16 *      5        3        1        0   \n"
        " *  17 *      7        6        4        2   \n"
        " #  24 #  *  21 *  *  20 *  *  19 *  *  18 * \n"
        " *  23 *     15       14       12       10   \n"
        " *  22 *     13       11        9        8   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.5, 0.5 }),
        "     0        1    *  16 *      5        4   \n"
        "     2        3    *  17 *      7        6   \n"
        " *  18 *  *  19 *  #  24 #  *  21 *  *  20 * \n"
        "    10       11    *  23 *     15       14   \n"
        "     8        9    *  22 *     13       12   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.9, 0.5 }),
        "     0        1        3        5    *  16 * \n"
        "     2        4        6        7    *  17 * \n"
        " *  18 *  *  19 *  *  20 *  *  21 *  #  24 # \n"
        "    10       12       14       15    *  23 * \n"
        "     8        9       11       13    *  22 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 2, 0.5 }),
        "     0        1        3        5        7   \n"
        "     2        4        6        8        9   \n"
        " *  20 *  *  21 *  *  22 *  *  23 *  *  24 * \n"
        "    12       14       16       18       19   \n"
        "    10       11       13       15       17   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { -1, 0.9 }),
        "    10        6        3        1        0   \n"
        "    14       11        7        4        2   \n"
        "    17       15       12        8        5   \n"
        "    19       18       16       13        9   \n"
        " *  24 *  *  23 *  *  22 *  *  21 *  *  20 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.1, 0.9 }),
        " *  16 *      6        3        1        0   \n"
        " *  17 *     10        7        4        2   \n"
        " *  18 *     13       11        8        5   \n"
        " *  19 *     15       14       12        9   \n"
        " #  24 #  *  23 *  *  22 *  *  21 *  *  20 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.5, 0.9 }),
        "     0        1    *  16 *      9        8   \n"
        "     2        3    *  17 *     11       10   \n"
        "     4        5    *  18 *     13       12   \n"
        "     6        7    *  19 *     15       14   \n"
        " *  20 *  *  21 *  #  24 #  *  23 *  *  22 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.9, 0.9 }),
        "     0        1        3        6    *  16 * \n"
        "     2        4        7       10    *  17 * \n"
        "     5        8       11       13    *  18 * \n"
        "     9       12       14       15    *  19 * \n"
        " *  20 *  *  21 *  *  22 *  *  23 *  #  24 # \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 2, 0.9 }),
        "     0        1        3        6       10   \n"
        "     2        4        7       11       14   \n"
        "     5        8       12       15       17   \n"
        "     9       13       16       18       19   \n"
        " *  20 *  *  21 *  *  22 *  *  23 *  *  24 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { -1, 2 }),
        "    10        6        3        1        0   \n"
        "    15       11        7        4        2   \n"
        "    19       16       12        8        5   \n"
        "    22       20       17       13        9   \n"
        "    24       23       21       18       14   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.1, 2 }),
        " *  20 *      6        3        1        0   \n"
        " *  21 *     10        7        4        2   \n"
        " *  22 *     14       11        8        5   \n"
        " *  23 *     17       15       12        9   \n"
        " *  24 *     19       18       16       13   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.5, 2 }),
        "     0        1    *  20 *     11       10   \n"
        "     2        3    *  21 *     13       12   \n"
        "     4        5    *  22 *     15       14   \n"
        "     6        7    *  23 *     17       16   \n"
        "     8        9    *  24 *     19       18   \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 0.9, 2 }),
        "     0        1        3        6    *  20 * \n"
        "     2        4        7       10    *  21 * \n"
        "     5        8       11       14    *  22 * \n"
        "     9       12       15       17    *  23 * \n"
        "    13       16       18       19    *  24 * \n");

    EXPECT_EQ(
        format_b2fo_result<2>(5, { 2, 2 }),
        "     0        1        3        6       10   \n"
        "     2        4        7       11       15   \n"
        "     5        8       12       16       19   \n"
        "     9       13       17       20       22   \n"
        "    14       18       21       23       24   \n");

    // --------

    /*
    Assertion below is generated using the following code,
    and then tested manually.

    std::cout << ft<2>(8, { 0.49, 0.49 }) << std::endl;
    */

    EXPECT_EQ(
        format_b2fo_result<2>(8, { 0.49, 0.49 }),
        "     0        1        3    *  49 *     15       12       10        9   \n"
        "     2        4        6    *  50 *     18       16       13       11   \n"
        "     5        7        8    *  51 *     20       19       17       14   \n"
        " *  52 *  *  53 *  *  54 *  #  63 #  *  58 *  *  57 *  *  56 *  *  55 * \n"
        "    29       31       32    *  62 *     48       47       45       42   \n"
        "    26       28       30    *  61 *     46       44       41       38   \n"
        "    23       25       27    *  60 *     43       40       37       35   \n"
        "    21       22       24    *  59 *     39       36       34       33   \n");
}

// Tests BackToFrontOrder<3>.
TEST(BackToFrontOrderTest, Basic_3d)
{
    /*
    Assertion below is generated using the following code,
    and then tested manually.

    std::cout << ft<3>(4, { 0.49, 0.49, 0.49 }) << std::endl;
    */

    EXPECT_EQ(
        format_b2fo_result<3>(4, { 0.49, 0.49, 0.49 }),
        "Plane 0:\n"
        "     0    *  27 *      2        1   \n"
        " *  28 *  #  54 #  *  30 *  *  29 * \n"
        "     4    *  32 *      8        7   \n"
        "     3    *  31 *      6        5   \n"
        "\n"
        "Plane 1:\n"
        " *  33 *  #  55 #  *  35 *  *  34 * \n"
        " #  56 #  @  63 @  #  58 #  #  57 # \n"
        " *  37 *  #  60 #  *  41 *  *  40 * \n"
        " *  36 *  #  59 #  *  39 *  *  38 * \n"
        "\n"
        "Plane 2:\n"
        "    10    *  43 *     14       13   \n"
        " *  45 *  #  62 #  *  49 *  *  48 * \n"
        "    18    *  53 *     26       25   \n"
        "    17    *  52 *     24       22   \n"
        "\n"
        "Plane 3:\n"
        "     9    *  42 *     12       11   \n"
        " *  44 *  #  61 #  *  47 *  *  46 * \n"
        "    16    *  51 *     23       21   \n"
        "    15    *  50 *     20       19   \n"
        "\n");

    // --------

    /*
    Assertion below is generated using the following code,
    and then tested manually.

    std::cout << ft<3>(4, { -1, -1, 2 }) << std::endl;
    */

    EXPECT_EQ(
        format_b2fo_result<3>(4, { -1, -1, 2 }),
        "Plane 0:\n"
        "    44       33       22       13   \n"
        "    32       21       12        6   \n"
        "    20       11        5        2   \n"
        "    10        4        1        0   \n"
        "\n"
        "Plane 1:\n"
        "    54       46       36       26   \n"
        "    45       35       25       16   \n"
        "    34       24       15        8   \n"
        "    23       14        7        3   \n"
        "\n"
        "Plane 2:\n"
        "    60       56       49       40   \n"
        "    55       48       39       29   \n"
        "    47       38       28       18   \n"
        "    37       27       17        9   \n"
        "\n"
        "Plane 3:\n"
        "    63       62       59       53   \n"
        "    61       58       52       43   \n"
        "    57       51       42       31   \n"
        "    50       41       30       19   \n"
        "\n");
}

} // anonymous namespace
