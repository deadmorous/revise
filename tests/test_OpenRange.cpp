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

#include "OpenRange.hpp"

#include <gtest/gtest.h>

namespace
{

using namespace s3dmm;


template <typename T>
class OpenRangeTest : public testing::Test {};

TYPED_TEST_SUITE_P(OpenRangeTest);

TYPED_TEST_P(OpenRangeTest, Basic)
{
    using T = TypeParam;
    using R = OpenRange<T>;

    auto empty_r = R::make_empty();
    EXPECT_TRUE(empty_r.empty());
    EXPECT_FALSE(empty_r.is_normalized());
    EXPECT_EQ(empty_r.normalized(), empty_r);

    auto empty_r2 = R{10, 10};
    EXPECT_TRUE(empty_r2.empty());
    EXPECT_FALSE(empty_r2.is_normalized());
    EXPECT_EQ(empty_r2.normalized(), empty_r2);

    EXPECT_EQ(empty_r, empty_r2);

    auto r1 = R{1, 5};
    EXPECT_FALSE(r1.empty());
    EXPECT_TRUE(r1.is_normalized());
    EXPECT_EQ(r1, r1.normalized());

    auto r2 = R{5, 1};
    EXPECT_FALSE(r2.empty());
    EXPECT_FALSE(r2.is_normalized());
    EXPECT_NE(r2, r2.normalized());
    EXPECT_EQ(r1, r2.normalized());

    r2.normalize();
    EXPECT_EQ(r1, r2);

    using Base = MultiIndex<2, T>;
    Base empty_r_base = empty_r;
    Base empty_r2_base = empty_r2;
    EXPECT_NE(empty_r_base, empty_r2_base);
    EXPECT_EQ(empty_r_base, empty_r);
    EXPECT_EQ(empty_r2_base, empty_r2);
}

REGISTER_TYPED_TEST_SUITE_P(OpenRangeTest, Basic);

using OpenRangeTypes = ::testing::Types<unsigned int, real_type>;
INSTANTIATE_TYPED_TEST_SUITE_P(Typed, OpenRangeTest, OpenRangeTypes);

} // anonymous namespace
