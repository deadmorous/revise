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

#include "s3dmm/Metadata.hpp"
#include <fstream>
#include <iostream>

#include <boost/iterator/iterator_facade.hpp>

#include "s3dmm_cuda/DeviceVector.hpp"
#include "s3dmm_cuda/DenseFieldInterpolator_cu_3.hpp"
#include "s3dmm/BlockTreeFieldService.hpp"

using namespace std;
using namespace s3dmm;

namespace s3dmm {
namespace detail {

template<unsigned int N>
inline ostream& operator<<(ostream& s, const TreeBlockId<N>& blockId) {
    s << blockId.index << '/' << blockId.level << '/' << blockId.location;
    return s;
}

} // detail
} // s3dmm

template <unsigned int N>
class BreadthFirstWalker
{
public:
    using BT = CompressedBlockTree<N>;
    using BlockId = typename BT::BlockId;
    using BC = CachedBitCounter<BT::ChildCount>;

    explicit BreadthFirstWalker(const BT& bt) :
        m_data(bt.data())
    {
        BC::maybeInit();
    }

    class iterator : public boost::iterator_facade<
            iterator, const BlockId, boost::forward_traversal_tag>
    {
    public:
        const BlockId& dereference() const {
            return m_currentBlockId;
        }

        void increment()
        {
            BOOST_ASSERT(m_data);   // Not currently at end
            if (m_currentBlockId.level == m_passedLevels.size()) {
                m_currentLevel.nodeCount += BC::countOnes(m_bp.get(m_currentLevel.currentPos));
                m_passedLevels.push_back(m_currentLevel);
                ++m_currentLevel.currentPos;
                m_currentLevel.startPos = m_currentLevel.currentPos;
                m_currentLevel.nodeCount = 0;
                for (auto& passedLevel : m_passedLevels)
                    passedLevel.currentPos = passedLevel.startPos;
            }
//            m_currentLevel.nodeCount +=
//            ++m_currentPos;
//            if (m_currentPos >= m_data->bitCount >> N) {
//                // End reached
//                m_data = nullptr;
//                return;
//            }
//            if (m_passedLevels.empty()) {
//                m_passedLevels.push_back(m_currentLevel);
//            }
//            else {
//                auto& lastLevel = m_passedLevels.back();
//                if (lastLevel.nodeCount == m_currentPos - m_currentLevel.startPos) {
//                    m_passedLevels.push_back(m_currentLevel);
//                    // TODO: compute current_pos at all passed levels
//                }
//                else {

//                }
//            }
        }

        bool equal(const iterator& that)
        {
            BOOST_ASSERT(m_data == nullptr || that.m_data == nullptr || m_data == that.m_data);
            return (m_data == nullptr && that.m_data == nullptr) || (m_currentBlockId == that.m_currentBlockId);
        }

    private:
        const typename BT::Data *m_data;
        BlockId m_currentBlockId;
        struct LevelData
        {
            std::size_t startPos = 0;
            std::size_t nodeCount = 0;
            std::size_t currentPos = 0;
        };
        std::vector<LevelData> m_passedLevels;
        LevelData m_currentLevel;
        ConstBitPacker_t<BT::ChildCount> m_bp;

        iterator() : m_data(nullptr), m_bp(nullptr)
        {
        }

        explicit iterator(const typename BT::Data& data) :
            m_data(&data), m_bp(data.data.data())
        {
        }

        friend class BreadthFirstWalker<N>;
    };

    iterator begin() const {
        return iterator(m_data);
    }

    iterator end() const {
        return iterator();
    }

private:
    const typename BT::Data& m_data;
};

template <unsigned int N>
BreadthFirstWalker<N> makeBreadthFirstWalker(const CompressedBlockTree<N>& bt) {
    return BreadthFirstWalker<N>(bt);
}

template <unsigned int N, class BT>
void printBlockTree(const BT& bt)
{
    using BC = CachedBitCounter<BT::ChildCount>;
    BC::maybeInit();
    auto& d1 = bt.data();
    auto bp = makeBitPacker<BT::ChildCount>(d1.data.data());
    auto nodeCount = d1.dataBitCount >> N;
    auto levelStartPos = 0u;
    auto levelEndPos = 1u;
    auto level = 0u;
    auto levelNodeCount = 0u;
    for (auto inode=0u; inode<nodeCount; ++inode) {
        if (inode == levelStartPos)
            cout << "level " << level << ": ";
        if (inode == levelEndPos) {
            cout << endl;
            levelStartPos = levelEndPos;
            levelEndPos += levelNodeCount;
            levelNodeCount = 0u;
            ++level;
            cout << "level " << level << ": ";
        }
        auto children = bp.get(inode);
        levelNodeCount += BC::countOnes(children);
        auto mask = 1 << (BT::ChildCount-1);
        cout << inode << ':';
        for (auto ichild=0u; ichild<BT::ChildCount; ++ichild, mask >>= 1)
            cout << (children & mask? '1': '0');
        cout << " ";
    }
    cout << endl;
}

template <unsigned int N>
void run(const string& inputFileName1, const string& inputFileName2, const string& outputFileName)
{
    using MD = Metadata<N>;
    using BT = typename MD::BT;
    auto openInputStream = [](const string& fileName) {
        ifstream is(fileName, ios::binary);
        if (is.fail())
            throw runtime_error(string("Failed to open input file '") + fileName + "'");
        is.exceptions(istream::failbit);
        return is;
    };
    auto is1 = openInputStream(inputFileName1);
    auto is2 = openInputStream(inputFileName2);
    MD md1(is1);
    MD md2(is2);
    auto& bt1 = md1.blockTree();
    auto& bt2 = md2.blockTree();
    // printBlockTree<N>(bt1);

    auto btw = makeBreadthFirstWalker(bt1);
    auto it = btw.begin();
    for (auto i=0u; i<100u; ++i) {
        cout << *it << endl;
        ++it;
    }
}

void test_merge(int argc, char *argv[])
{
    if (argc != 5)
        throw runtime_error("Usage: s3dmm_prep dim input_metadata_1 input_metadata_2 output_metadata ");
    auto N = static_cast<unsigned int>(atoi(argv[1]));
    switch (N) {
    case 1:
        run<1>(argv[2], argv[3], argv[4]);
        break;
    case 2:
        run<2>(argv[2], argv[3], argv[4]);
        break;
    case 3:
        run<3>(argv[2], argv[3], argv[4]);
        break;
    default:
        throw range_error("Invalid space dimension (must be 1, 2, or 3)");
    }
}

#ifdef S3DMM_ENABLE_CUDA
void test_cuda(int argc, char *argv[])
{
//    auto openInputStream = [](const string& fileName) {
//        ifstream is(fileName, ios::binary);
//        if (is.fail())
//            throw runtime_error(string("Failed to open input file '") + fileName + "'");
//        is.exceptions(istream::failbit);
//        return is;
//    };

    if (argc != 3)
        throw runtime_error("Usage: s3dmm_prep input_mesh field");
    string baseName = argv[1];
//    string fieldName = argv[2];
//    auto mdFileName = baseName + ".s3dmm-meta";
//    auto mdStream = openInputStream(mdFileName);
//    Metadata<3> md(mdStream);
    BlockTreeFieldService<3> fieldSvc(baseName);

//    auto is1 = openInputStream(inputFileName1);
//    auto is2 = openInputStream(inputFileName2);
//    MD md1(is1);


    DeviceVector<int> v({1,2,3});
    auto x = v.download();
    int a = 123;
}

void test_interp_cuda(int argc, char *argv[])
{
    if (argc != 3)
        throw runtime_error("Usage: s3dmm_prep input_mesh field");
    string baseName = argv[1];
    BlockTreeFieldService<3> fieldSvc(baseName);
    DeviceVector<dfield_real> dfield(fieldSvc.denseFieldSize());
    using BlockId = Metadata<3>::BlockId;
    Vec2<dfield_real> fieldRange;
    fieldSvc.interpolateWith<DenseFieldInterpolator_cu_3>(
                fieldRange,
                dfield.data(), 0 /*TODO: fieldIndex*/, 0, BlockId());
    auto f = dfield.download();

    // deBUG, TODO: Remove
    vector<dfield_real> f0;
    fieldSvc.interpolate(fieldRange, f0, 0, 0, BlockId());

    auto isnanLambda = [](auto x) { return isnan(x); };
    auto nnan = count_if(f.begin(), f.end(), isnanLambda);
    auto nnan0 = count_if(f0.begin(), f0.end(), isnanLambda);

    int a = 123;
}
#else
void test_cuda(int /*argc*/, char */*argv*/[]) {}
void test_interp_cuda(int /*argc*/, char */*argv*/[]) {}
#endif // S3DMM_ENABLE_CUDA

int main(int argc, char *argv[])
{
    try {
        // test_merge(argc, argv);
        // test_cuda(argc, argv);
        test_interp_cuda(argc, argv);

        return EXIT_SUCCESS;
    }
    catch(const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}
