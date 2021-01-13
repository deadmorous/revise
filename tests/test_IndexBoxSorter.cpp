#include "gtest/gtest.h"

#include "IndexBoxSorter.hpp"
#include "IndexCubeSplitter.hpp"

#include <string>

namespace test_s3dmm
{

using namespace s3dmm;

using BBox = IndexBoxSorter::BBox;

TEST(TestIndexBoxSorter, testEqualToOriginal)
{
    std::vector<BBox> boxes;
    for (unsigned int i = 0; i < 10; i++)
    {
        Vec3u v{i, 0, 0};
        BBox box;
        box << v << v + Vec3u{1, 10, 10};
        boxes.push_back(box);
    }

    auto testEqualToOrig = [&boxes](const Vec3d& dir, bool oblique = false) {
        IndexBoxSorter sorter;
        sorter.setDirection(dir);
        sorter.setBoxes(boxes);
        auto res = sorter.getBoxesSorted();
        ASSERT_TRUE(res.size() == boxes.size());
        for (size_t i = 0; i < res.size(); i++)
        {
            auto i2 = oblique ? (res.size() - 1 - i) : i;
            EXPECT_TRUE(res[i].max() == boxes[i2].max());
            EXPECT_TRUE(res[i].min() == boxes[i2].min());
        }
    };

    testEqualToOrig({1, 0, 0});
    testEqualToOrig({1, 1, 0});
    testEqualToOrig({1, 1, 1});

    testEqualToOrig({-1, 0, 0}, true);
    testEqualToOrig({-1, -1, 0}, true);
    testEqualToOrig({-1, -1, -1}, true);
}

TEST(TestIndexBoxSorter, testCubeRegularSplits)
{
    auto testSplit = [](unsigned int level,
                        unsigned int splitCount,
                        const Vec3d& dir) {
        IndexCubeSplitter splitter;
        auto boxes = splitter.split(level, splitCount);
        IndexBoxSorter sorter;
        auto res = sorter.getBoxesSorted(boxes, dir);

        ASSERT_TRUE(res.size() == boxes.size());

        auto boxesSorted = boxes;
        std::sort(boxesSorted.begin(), boxesSorted.end(), [&](auto a, auto b) {
            Vec3d ra = elementwiseMultiply(a.max(), Vec3d{1, 1, 1});
            Vec3d rb = elementwiseMultiply(b.max(), Vec3d{1, 1, 1});
            return ra * dir < rb * dir;
        });

        auto compBox = [](auto a, auto b) {
            return a.max() == b.max() && a.min() == b.min();
        };

        // We can compare only the first and the last elements,
        // because IndexBoxSorter does not provide strict sorting
        // according to the distance along dir
        if (!res.empty())
        {
            EXPECT_TRUE(compBox(res.front(), boxesSorted.front()));
            EXPECT_TRUE(compBox(res.back(), boxesSorted.back()));
        }
    };

    testSplit(0, 1, {1, 2, 3});

    testSplit(1, 8, {1, 2, 3});

    testSplit(2, 8, {1, 2, 3});
    testSplit(2, 8, {1, 2, -3});

    testSplit(2, 16, {1, 2, 3});
    testSplit(2, 64, {1, 2, 3});
    testSplit(2, 64, {-1, -2, -3});
}

TEST(TestIndexBoxSorter, testSimpleHandmadeSeq)
{
    BBox box1;
    box1 << Vec3u{1, 0, 0} << Vec3u{2, 4, 1};
    BBox box2;
    box2 << Vec3u{0, 3, 0} << Vec3u{1, 4, 1};
    BBox box3;
    box3 << Vec3u{2, 0, 0} << Vec3u{3, 1, 1};
    IndexBoxSorter sorter;
    auto res = sorter.getBoxesSorted({box1, box2, box3}, {1, 1, 0});

    auto compBox = [](auto a, auto b) {
        return a.max() == b.max() && a.min() == b.min();
    };

    EXPECT_TRUE(compBox(res[0], box2));
    EXPECT_TRUE(compBox(res[1], box1));
    EXPECT_TRUE(compBox(res[2], box3));
}


} // namespace test_s3dmm
