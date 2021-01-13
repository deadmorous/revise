#include "gtest/gtest.h"

#include "BlockSplitter.hpp"

#include <string>

namespace test_s3dmm
{

using namespace s3dmm;

TEST(TestBlockSplitter, DISABLED_testBlockSplitter)
{
    Vec3d dir{1., 2., 3.};
    Vec3d e0{1., 0, 0}, e1{0, 1., 0};

    auto doSplit = [dir, e0, e1](unsigned int level, unsigned int splitCount) {
        BlockSplitter bs;
        auto res = bs.split(level, splitCount, dir, e0, e1);
        EXPECT_TRUE(res.size() == splitCount);
        return res;
    };

    {
        auto res = doSplit(0, 1);

        auto item = res[0];
        EXPECT_TRUE(item.size() == 1);
    }

    {
        auto res = doSplit(1, 2);

        auto item = res[0];
        EXPECT_TRUE(item.size() == 1);

        item = res[1];
        EXPECT_TRUE(item.size() == 1);
    }
}

} // namespace test_s3dmm
