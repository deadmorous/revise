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

#include "NonUniformIndexRangeSplitter.hpp"
#include "UniformIndexRangeSplitter.hpp"

#include <gtest/gtest.h>

namespace
{

using namespace s3dmm;

TEST(UniformIndexRangeSplitterTest, Basic)
{
    using IR = IndexRange;
    auto splitter = UniformIndexRangeSplitter{{10, 20}, {100, 101}};

    auto empty_r = IndexRange::make_empty();
    EXPECT_EQ(splitter(  0.  ), IndexRangeSplit( empty_r,   ~0u, IR{10, 20} ));
    EXPECT_EQ(splitter( 99.99), IndexRangeSplit( empty_r,   ~0u, IR{10, 20} ));
    EXPECT_EQ(splitter(100.01), IndexRangeSplit( empty_r,   10u, IR{11, 20} ));
    EXPECT_EQ(splitter(100.02), IndexRangeSplit( empty_r,   10u, IR{11, 20} ));
    EXPECT_EQ(splitter(100.07), IndexRangeSplit( empty_r,   10u, IR{11, 20} ));
    EXPECT_EQ(splitter(100.09), IndexRangeSplit( empty_r,   10u, IR{11, 20} ));
    EXPECT_EQ(splitter(100.11), IndexRangeSplit( IR{10,11}, 11u, IR{12, 20} ));
    EXPECT_EQ(splitter(100.15), IndexRangeSplit( IR{10,11}, 11u, IR{12, 20} ));
    EXPECT_EQ(splitter(100.19), IndexRangeSplit( IR{10,11}, 11u, IR{12, 20} ));
    EXPECT_EQ(splitter(100.21), IndexRangeSplit( IR{10,12}, 12u, IR{13, 20} ));
    EXPECT_EQ(splitter(100.25), IndexRangeSplit( IR{10,12}, 12u, IR{13, 20} ));
    EXPECT_EQ(splitter(100.29), IndexRangeSplit( IR{10,12}, 12u, IR{13, 20} ));
    EXPECT_EQ(splitter(100.31), IndexRangeSplit( IR{10,13}, 13u, IR{14, 20} ));
    EXPECT_EQ(splitter(100.39), IndexRangeSplit( IR{10,13}, 13u, IR{14, 20} ));
    EXPECT_EQ(splitter(100.41), IndexRangeSplit( IR{10,14}, 14u, IR{15, 20} ));
    EXPECT_EQ(splitter(100.49), IndexRangeSplit( IR{10,14}, 14u, IR{15, 20} ));
    EXPECT_EQ(splitter(100.51), IndexRangeSplit( IR{10,15}, 15u, IR{16, 20} ));
    EXPECT_EQ(splitter(100.59), IndexRangeSplit( IR{10,15}, 15u, IR{16, 20} ));
    EXPECT_EQ(splitter(100.61), IndexRangeSplit( IR{10,16}, 16u, IR{17, 20} ));
    EXPECT_EQ(splitter(100.69), IndexRangeSplit( IR{10,16}, 16u, IR{17, 20} ));
    EXPECT_EQ(splitter(100.71), IndexRangeSplit( IR{10,17}, 17u, IR{18, 20} ));
    EXPECT_EQ(splitter(100.79), IndexRangeSplit( IR{10,17}, 17u, IR{18, 20} ));
    EXPECT_EQ(splitter(100.81), IndexRangeSplit( IR{10,18}, 18u, IR{19, 20} ));
    EXPECT_EQ(splitter(100.89), IndexRangeSplit( IR{10,18}, 18u, IR{19, 20} ));
    EXPECT_EQ(splitter(100.91), IndexRangeSplit( IR{10,19}, 19u, empty_r    ));
    EXPECT_EQ(splitter(100.99), IndexRangeSplit( IR{10,19}, 19u, empty_r    ));
    EXPECT_EQ(splitter(101.01), IndexRangeSplit( IR{10,20}, ~0u, empty_r    ));
    EXPECT_EQ(splitter(999.99), IndexRangeSplit( IR{10,20}, ~0u, empty_r    ));

    ASSERT_DEATH(
        { UniformIndexRangeSplitter({20, 10}, {100, 101}); },
        ".*Assertion `m_index_range\\.is_normalized\\(\\)' failed\\." );

    ASSERT_DEATH(
        { UniformIndexRangeSplitter({10, 20}, {101, 100}); },
        ".*Assertion `m_coord_range\\.is_normalized\\(\\)' failed\\." );
}

TEST(NonUniformIndexRangeSplitterTest, Basic)
{
    using IR = IndexRange;
    real_type split_coords[] = {102, 105, 109};
    auto splitter = NonUniformIndexRangeSplitter{5, split_coords, {100, 110}};

    auto empty_r = IndexRange::make_empty();
    EXPECT_EQ(splitter(  0.  ), IndexRangeSplit( empty_r,  ~0u, IR{5, 9} ));
    EXPECT_EQ(splitter( 99.99), IndexRangeSplit( empty_r,  ~0u, IR{5, 9} ));
    EXPECT_EQ(splitter(100.01), IndexRangeSplit( empty_r,   5u, IR{6, 9} ));
    EXPECT_EQ(splitter(101.  ), IndexRangeSplit( empty_r,   5u, IR{6, 9} ));
    EXPECT_EQ(splitter(101.99), IndexRangeSplit( empty_r,   5u, IR{6, 9} ));
    EXPECT_EQ(splitter(102.01), IndexRangeSplit( IR{5, 6},  6u, IR{7, 9} ));
    EXPECT_EQ(splitter(103.  ), IndexRangeSplit( IR{5, 6},  6u, IR{7, 9} ));
    EXPECT_EQ(splitter(104.  ), IndexRangeSplit( IR{5, 6},  6u, IR{7, 9} ));
    EXPECT_EQ(splitter(104.99), IndexRangeSplit( IR{5, 6},  6u, IR{7, 9} ));
    EXPECT_EQ(splitter(105.01), IndexRangeSplit( IR{5, 7},  7u, IR{8, 9} ));
    EXPECT_EQ(splitter(106.  ), IndexRangeSplit( IR{5, 7},  7u, IR{8, 9} ));
    EXPECT_EQ(splitter(107.  ), IndexRangeSplit( IR{5, 7},  7u, IR{8, 9} ));
    EXPECT_EQ(splitter(108.  ), IndexRangeSplit( IR{5, 7},  7u, IR{8, 9} ));
    EXPECT_EQ(splitter(108.99), IndexRangeSplit( IR{5, 7},  7u, IR{8, 9} ));
    EXPECT_EQ(splitter(109.01), IndexRangeSplit( IR{5, 8},  8u, empty_r  ));
    EXPECT_EQ(splitter(109.99), IndexRangeSplit( IR{5, 8},  8u, empty_r  ));
    EXPECT_EQ(splitter(110.01), IndexRangeSplit( IR{5, 9}, ~0u, empty_r  ));
    EXPECT_EQ(splitter(100000), IndexRangeSplit( IR{5, 9}, ~0u, empty_r  ));

    ASSERT_DEATH(
        { NonUniformIndexRangeSplitter(5, split_coords, {110, 100}); },
        ".*Assertion `coord_range\\.is_normalized\\(\\)' failed\\.");

    ASSERT_DEATH(
        { NonUniformIndexRangeSplitter(5, split_coords, {103, 110}); },
        ".*Assertion `std::is_sorted\\(m_coords.begin\\(\\),"
        " m_coords.end\\(\\)\\)' failed\\.");
}

} // anonymous namespace
