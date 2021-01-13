#include "gtest/gtest.h"

#include "BlockSorter.hpp"

#include <string>

namespace test_s3dmm
{

using namespace s3dmm;

TEST(TestBlockSorter, testBlockSorter)
{
    Vec3d dir{1., 2., 3.};

    auto testSortBlocks = [dir](
                              const std::vector<BlockSorter::BBox>& boxes,
                              const std::vector<Vec3u>& expected) {
        auto res = BlockSorter::sortBlocks(boxes, dir);
        ASSERT_TRUE(res.size() == expected.size());
        for (size_t i = 0; i < res.size(); i++)
        {
            EXPECT_TRUE(res[i] == expected[i]);
        }
    };

    {
        BlockSorter::BBox box;
        box << Vec3u{0, 0, 0} << Vec3u{1, 1, 1};
        testSortBlocks({{box}}, {{0, 0, 0}});
    }

    {
        BlockSorter::BBox box;
        box << Vec3u{0, 0, 0} << Vec3u{2, 2, 2};
        testSortBlocks(
            {{box}},
            {{0, 0, 0},
             {1, 0, 0},
             {0, 1, 0},
             {1, 1, 0},
             {0, 0, 1},
             {1, 0, 1},
             {0, 1, 1},
             {1, 1, 1}});
    }

    {
        BlockSorter::BBox box;
        testSortBlocks({box, box}, std::vector<Vec3u>());
    }
}

} // namespace test_s3dmm
