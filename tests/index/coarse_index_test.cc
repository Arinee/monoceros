#include "index/coarse_index.h"
#include <cmath>
#include <stdlib.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mercury;
using namespace std;

class CoarseIndexTest: public testing::Test
{
public:
    void SetUp()
    {
        _pBase = new char[40960];
        memset(_pBase, 0, sizeof(char)*40960);
        _slotNum = 5;
        _maxDocSize = 100;
        CoarseIndex *index = new CoarseIndex();
        bool bret = index->create(_pBase, 40960, _slotNum, _maxDocSize);
        ASSERT_EQ(true, bret);
        delete index;
    }

    void TearDown()
    {
        delete []_pBase;
    }

    char *_pBase;
    int64_t _maxDocSize;
    int _slotNum;
};

TEST_F(CoarseIndexTest, TestCalcSize) 
{
    int64_t maxDocSize = 1000;
    int slotNum = 10;
    int blockCnt = slotNum;
    size_t expectCapacity = blockCnt * sizeof(Block) + slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    EXPECT_EQ(expectCapacity, CoarseIndex::calcSize(slotNum, maxDocSize));

    maxDocSize = 100000;
    slotNum = 10;
    blockCnt = (size_t)ceil(1.0 * maxDocSize / DOCS_PER_BLOCK) + slotNum;
    expectCapacity =  blockCnt * sizeof(Block) + slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    EXPECT_EQ(expectCapacity, CoarseIndex::calcSize(slotNum, maxDocSize));
}

TEST_F(CoarseIndexTest, TestCreate)
{
    char *pBase = new char[40960];
    memset(pBase, 0, sizeof(char)*40960);
    CoarseIndex *index = new CoarseIndex();
    int64_t slotNum = 5;
    int64_t maxDocSize = 100;
    bool bret = index->create(pBase, 40960, slotNum, maxDocSize);
    EXPECT_EQ(true, bret);

    auto header = index->getHeader();
    EXPECT_EQ(slotNum, header->slotNum);
    EXPECT_EQ(maxDocSize, header->maxDocSize);
    int64_t expectCapacity = slotNum * sizeof(Block) + slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    EXPECT_EQ(expectCapacity, header->capacity);
    int64_t expectUsedSize = sizeof(CoarseIndex::Header) + slotNum * sizeof(IndexSlot);
    EXPECT_EQ(expectUsedSize, header->usedSize);
    IndexSlot *expectIndexSlot = (IndexSlot*)(pBase + sizeof(CoarseIndex::Header));
    EXPECT_EQ(expectIndexSlot, index->getIndexSlot());
    Block *expectBlock = (Block*)(pBase + sizeof(CoarseIndex::Header) + sizeof(IndexSlot)*slotNum);
    EXPECT_EQ(expectBlock, index->getBlockPosting());

    delete index;
    delete []pBase;
}

TEST_F(CoarseIndexTest, TestLoad)
{
    char *pBase = new char[40960];
    memset(pBase, 0, sizeof(char)*40960);
    CoarseIndex *index = new CoarseIndex();
    int64_t slotNum = 5;
    int64_t maxDocSize = 100;
    bool bret = index->create(pBase, 40960, slotNum, maxDocSize);
    EXPECT_EQ(true, bret);
    delete index;
    index = nullptr;

    CoarseIndex *loadIndex = new CoarseIndex();
    int64_t blockCnt = slotNum;
    int64_t expectCapacity = blockCnt * sizeof(Block) + slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    bret = loadIndex->load(pBase, expectCapacity);
    EXPECT_EQ(true, bret);

    auto header = loadIndex->getHeader();
    EXPECT_EQ(slotNum, header->slotNum);
    EXPECT_EQ(maxDocSize, header->maxDocSize);
    EXPECT_EQ(expectCapacity, header->capacity);
    int64_t expectUsedSize = sizeof(CoarseIndex::Header) + slotNum * sizeof(IndexSlot);
    EXPECT_EQ(expectUsedSize, header->usedSize);
    IndexSlot *expectIndexSlot = (IndexSlot*)(pBase + sizeof(CoarseIndex::Header));
    EXPECT_EQ(expectIndexSlot, loadIndex->getIndexSlot());
    Block *expectBlock = (Block*)(pBase + sizeof(CoarseIndex::Header) + sizeof(IndexSlot)*slotNum);
    EXPECT_EQ(expectBlock, loadIndex->getBlockPosting());

    delete loadIndex;
    delete []pBase;
}

TEST_F(CoarseIndexTest, TestAddDoc1)
{
    CoarseIndex *index= new CoarseIndex();
    int64_t expectCapacity = _slotNum * sizeof(Block) + _slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    bool bret = index->load(_pBase, expectCapacity);
    EXPECT_EQ(true, bret);

    docid_t docId = 1;
    int32_t coarseLabel = 1;

    bret = index->addDoc(coarseLabel, docId);
    EXPECT_EQ(true, bret);
    auto pIndexSlot = index->getIndexSlot();
    auto slot = pIndexSlot + coarseLabel;
    EXPECT_EQ(sizeof(CoarseIndex::Header) + _slotNum * sizeof(IndexSlot), (uint64_t)slot->offset);
    EXPECT_EQ(sizeof(CoarseIndex::Header) + _slotNum * sizeof(IndexSlot), (uint64_t)slot->lastOffset);
    auto block = index->getBlockPosting();
    EXPECT_EQ(1L, block->next);
    EXPECT_EQ(docId, block->attr[0].docId);

    delete index;
}

TEST_F(CoarseIndexTest, TestDump) 
{
    CoarseIndex *index = new CoarseIndex();
    int64_t expectLen = _slotNum * sizeof(Block) + _slotNum * sizeof(IndexSlot) + sizeof(CoarseIndex::Header);
    bool bret = index->load(_pBase, expectLen);
    EXPECT_EQ(true, bret);

    docid_t docId1 = 1;
    int32_t coarseLabel1 = 1;
    bret = index->addDoc(docId1, coarseLabel1);
    EXPECT_EQ(true, bret);

    docid_t docId2 = 2;
    int32_t coarseLabel2 = 2;
    bret = index->addDoc(docId2, coarseLabel2);
    EXPECT_EQ(true, bret);

    string dumpFile = "coarse_index.dat";
    bret = index->dump(dumpFile);
    EXPECT_EQ(true, bret);

    FILE *fp = fopen(dumpFile.c_str(), "r");
    ASSERT_TRUE(fp != NULL);
    struct stat tmpStat;
    int ret = fstat(fileno(fp), &tmpStat);
    ASSERT_EQ(0, ret);
    int64_t fileSize = tmpStat.st_size;
    EXPECT_EQ(expectLen, fileSize);
    char *tmpBuf = new char[40960];
    int readLen = fread(tmpBuf, sizeof(char), fileSize, fp);
    EXPECT_EQ(fileSize, readLen);

    CoarseIndex *dumpIndex = new CoarseIndex();
    bret = dumpIndex->load(tmpBuf, readLen);
    EXPECT_EQ(true, bret);

    auto iter = dumpIndex->search(coarseLabel1);
    EXPECT_EQ(docId1, iter.next());
    EXPECT_EQ(true, iter.finish());

    iter = dumpIndex->search(coarseLabel2);
    EXPECT_EQ(docId2, iter.next());
    EXPECT_EQ(true, iter.finish());

    delete dumpIndex;
    delete []tmpBuf;
    delete index;
    string cmd = string("rm -f ") + dumpFile;
    system(cmd.c_str());
}
