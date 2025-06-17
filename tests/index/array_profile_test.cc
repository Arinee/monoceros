#include <stdlib.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define protected public
#define private public
#include "index/array_profile.h"
#undef protected
#undef private

using namespace mercury;
using namespace std;

class ArrayProfileTest: public testing::Test
{
public:
    void SetUp()
    {
        _len = 40960;
        _base = new char[_len];
        memset(_base, 0, _len);
        _infoSize = sizeof(float) * 4; // float, dimension = 4
        _maxDocSize = (_len - sizeof(ArrayProfile::Header)) / _infoSize;
    }

    void TearDown()
    {
        delete []_base;
    }

    char *_base;
    int64_t _len;
    int64_t _maxDocSize;
    int64_t _infoSize;
};

TEST_F(ArrayProfileTest, TestCalcSize) 
{
    size_t expectCapacity = _maxDocSize * _infoSize + sizeof(ArrayProfile::Header);
    EXPECT_EQ(expectCapacity, ArrayProfile::CalcSize(_maxDocSize, _infoSize));

    int64_t maxDocSize = 1;
    int64_t infoSize = 5;
    expectCapacity = maxDocSize * infoSize + sizeof(ArrayProfile::Header);
    EXPECT_EQ(expectCapacity, ArrayProfile::CalcSize(maxDocSize, infoSize));
}

TEST_F(ArrayProfileTest, TestCreate)
{
    ArrayProfile *profile = new ArrayProfile();
    EXPECT_TRUE(profile->create(_base, _len, _infoSize));

    auto header = profile->getHeader();
    int64_t expectCapacity = _maxDocSize * _infoSize + sizeof(ArrayProfile::Header);
    EXPECT_EQ(expectCapacity, header->capacity);
    EXPECT_EQ(0L, header->usedDocNum);
    EXPECT_EQ(_maxDocSize, header->maxDocNum);

    EXPECT_EQ((void*)(_base+sizeof(ArrayProfile::Header)), profile->_infos);

    delete profile;
}

TEST_F(ArrayProfileTest, TestLoad)
{
    ArrayProfile *profile = new ArrayProfile();
    EXPECT_TRUE(profile->create(_base, _len, _infoSize));
    delete profile;
    profile = nullptr;

    ArrayProfile *failedLoad = new ArrayProfile();
    EXPECT_FALSE(failedLoad->load(_base, 64L));
    delete failedLoad;
    failedLoad = nullptr;

    ArrayProfile *loadProfile = new ArrayProfile();
    int64_t expectLen = _maxDocSize * _infoSize + sizeof(ArrayProfile::Header);
    EXPECT_TRUE(loadProfile->load(_base, expectLen));

    auto header = loadProfile->getHeader();
    auto values = loadProfile->_infos;
    EXPECT_EQ(expectLen, header->capacity);
    EXPECT_EQ(0L, header->usedDocNum);
    EXPECT_EQ(_maxDocSize, header->maxDocNum);
    EXPECT_EQ((void*)(_base+sizeof(ArrayProfile::Header)), values);

    delete loadProfile;
    loadProfile = nullptr;
}

TEST_F(ArrayProfileTest, TestInsertFeature)
{
    ArrayProfile *profile = new ArrayProfile();
    EXPECT_TRUE(profile->create(_base, _len, _infoSize));

    docid_t docid = 0;
    float info[] = {1.0,2.0,3.0,4.0};
    EXPECT_TRUE(profile->insert(docid, info));
    auto pInfo = (float*)profile->getInfo(docid);
    EXPECT_EQ(info[0], pInfo[0]);
    EXPECT_EQ(info[1], pInfo[1]);
    EXPECT_EQ(info[2], pInfo[2]);
    EXPECT_EQ(info[3], pInfo[3]);

    delete profile;
}

TEST_F(ArrayProfileTest, TestDump) 
{
    ArrayProfile *profile = new ArrayProfile();
    EXPECT_TRUE(profile->create(_base, _len, _infoSize));

    docid_t docid1 = 0;
    float feature1[] = {1.0,1.0,1.0,1.0};
    EXPECT_TRUE(profile->insert(docid1, feature1));

    docid_t docid2 = 1;
    float feature2[] = {2.0,2.0,2.0,2.0};
    EXPECT_TRUE(profile->insert(docid2, feature2));

    docid_t docid3 = 2;
    float feature3[] = {3.0,3.0,3.0,3.0};
    EXPECT_TRUE(profile->insert(docid3, feature3));

    string dumpFile = "product_profile.dat";
    EXPECT_TRUE(profile->dump(dumpFile));

    FILE *fp = fopen(dumpFile.c_str(), "r");
    ASSERT_TRUE(fp != NULL);
    struct stat tmpStat;
    int ret = fstat(fileno(fp), &tmpStat);
    ASSERT_EQ(0, ret);
    int64_t fileSize = tmpStat.st_size;
    int64_t expectLen = _maxDocSize * _infoSize + sizeof(ArrayProfile::Header);
    EXPECT_EQ(expectLen, fileSize);
    char *tmpBuf = new char[40960];
    int readLen = fread(tmpBuf, sizeof(char), fileSize, fp);
    EXPECT_EQ(fileSize, readLen);

    ArrayProfile *dumpProfile = new ArrayProfile();
    EXPECT_TRUE(dumpProfile->load(tmpBuf, readLen));

    auto pInfo = (float*)dumpProfile->getInfo(docid1);
    EXPECT_EQ(feature1[0], pInfo[0]);

    pInfo = (float*)dumpProfile->getInfo(docid2);
    EXPECT_EQ(feature2[0], pInfo[0]);

    pInfo = (float*)dumpProfile->getInfo(docid3);
    EXPECT_EQ(feature3[0], pInfo[0]);

    delete dumpProfile;
    delete []tmpBuf;
    delete profile;
    string cmd = string("rm -f ") + dumpFile;
    system(cmd.c_str());
}

