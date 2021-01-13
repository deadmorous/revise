#include "gtest/gtest.h"

#include "IndexCubeSplitter.hpp"

#include <string>

namespace test_s3dmm
{

using namespace s3dmm;

using BBox = IndexCubeSplitter::BBox;

bool unionBoundingBox(BBox& box1, const BBox& box2)
{
    if (box2.empty())
    {
        return true;
    }
    if (box1.empty())
    {
        box1 = box2;
        return true;
    }

    auto isRangeEqual = [&](unsigned int idx) {
        auto result = (box1.max()[idx] == box2.max()[idx])
                      && (box1.min()[idx] == box2.min()[idx]);
        return result;
    };

    for (unsigned int i = 0; i < 3; i++)
    {
        auto i1 = (i + 1) % 3;
        auto i2 = (i + 2) % 3;
        if (!isRangeEqual(i1) || !isRangeEqual(i2))
        {
            continue;
        }
        if (box1.max()[i] == box2.min()[i])
        {
            box1 << box2.max();
            return true;
        }
        if (box1.min()[i] == box2.max()[i])
        {
            box1 << box2.min();
            return true;
        }
    }
    return false;
}

class TestIndexCubeSplitter : public ::testing::Test
{
public:
    void unionAllBoxes(std::vector<BBox>& boxes)
    {
        for (;;)
        {
            bool united = false;
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                for (size_t j = i + 1; j < boxes.size();)
                {
                    if (unionBoundingBox(boxes[i], boxes[j]))
                    {
                        united = true;
                        boxes.erase(boxes.begin() + j);
                    }
                    else
                    {
                        j++;
                    }
                }
            }
            if (!united)
            {
                break;
            }
        }
    }

    void testSplit(unsigned int level, unsigned int splitCount)
    {
        IndexCubeSplitter splitter;
        auto res = splitter.split(level, splitCount);
        ASSERT_TRUE(res.size() == splitCount);

        unionAllBoxes(res);
        ASSERT_TRUE(res.size() == 1);

        BBox boxFull;
        boxFull << Vec3u{0, 0, 0}
                << Vec3u{1u << level, 1u << level, 1u << level};

        EXPECT_TRUE(boxFull.min() == res[0].min());
        EXPECT_TRUE(boxFull.max() == res[0].max());
    }
};

TEST_F(TestIndexCubeSplitter, testUnionBoundingBox)
{
    {
        BBox b1, b2;

        EXPECT_TRUE(unionBoundingBox(b1, b2));
        EXPECT_TRUE(b1.empty());

        b1 << Vec3u{0, 0, 0} << Vec3u{2, 2, 2};
        b2 << Vec3u{2, 0, 0} << Vec3u{4, 2, 2};

        EXPECT_TRUE(unionBoundingBox(b1, b2));
        EXPECT_TRUE(b1.min() == Vec3u({0, 0, 0}));
        EXPECT_TRUE(b1.max() == b2.max());
    }

    {
        BBox b1, b2;
        b1 << Vec3u{0, 0, 0} << Vec3u{2, 2, 2};
        b2 << Vec3u{1, 0, 0} << Vec3u{4, 2, 2};
        EXPECT_FALSE(unionBoundingBox(b1, b2));
    }

    {
        BBox b1, b2;
        b1 << Vec3u{0, 0, 0} << Vec3u{2, 2, 2};
        b2 << Vec3u{0, 2, 0};
        EXPECT_FALSE(unionBoundingBox(b1, b2));

        b2 << Vec3u{2, 3, 2};
        EXPECT_TRUE(unionBoundingBox(b1, b2));
    }
}

TEST_F(TestIndexCubeSplitter, testSplitConsistency)
{
    testSplit(0, 1);
    testSplit(0, 2);

    testSplit(2, 2);
    testSplit(2, 3);

    testSplit(2, 5);
    testSplit(2, 9);

    testSplit(2, 64);
    testSplit(2, 100);

    testSplit(3, 32);
    testSplit(3, 64);
    testSplit(3, 512);

    testSplit(4, 1024);
}

TEST_F(TestIndexCubeSplitter, testSplitEquality)
{
    auto testIt = [](unsigned int level, unsigned int splitCount) {
        IndexCubeSplitter splitter;
        auto res = splitter.split(level, splitCount);
        EXPECT_TRUE(std::all_of(res.begin(), res.end(), [&](auto x) {
            return x.size() == res.front().size();
        }));
    };

    // It should be right when splitCount is the power of 2
    testIt(2, 2);
    testIt(2, 4);
    testIt(2, 8);
    testIt(2, 16);
    testIt(2, 32);
    testIt(2, 64);

    testIt(0, 1);

    testIt(1, 1);
    testIt(1, 2);
    testIt(1, 4);
    testIt(1, 8);

    testIt(3, 32);
    testIt(3, 64);
    testIt(3, 512);

    testIt(4, 1024);
}

} // namespace test_s3dmm
