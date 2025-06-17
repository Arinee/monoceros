#include <fstream>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "index/centroid_resource.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class CentroidResourceTest : public testing::Test
{
public:
    void SetUp()
    {
    }
    
    void TearDown()
    {
    }
};

TEST_F(CentroidResourceTest, TestCreate)
{
    CentroidResource cr;
    CentroidResource::RoughMeta rMeta(1024, 2, {100, 10}); //elemSize, levelCnt, centroidNumPerLevel
    CentroidResource::IntegrateMeta iMeta(16, 64, 100); //elemSize, fragmentNum, centroidNum
    ASSERT_TRUE(cr.create(rMeta, iMeta));

    CentroidResource::RoughMeta rMeta2(1024, 2, {100});
    ASSERT_FALSE(cr.create(rMeta2, iMeta));
}

TEST_F(CentroidResourceTest, TestDump)
{
    CentroidResource cr;
    CentroidResource::RoughMeta rMeta(4, 2, {2, 2});
    CentroidResource::IntegrateMeta iMeta(2, 2, 2);
    ASSERT_TRUE(cr.create(rMeta, iMeta));
    ASSERT_EQ(cr._roughMatrixSize, size_t((2+2*2)*4));
    ASSERT_EQ(cr._integrateMatrixSize, size_t(2*2*2));

    int32_t val = 1;
    cr.setValueInRoughMatrix(0, 0, (char*)&val);
    val++;
    cr.setValueInRoughMatrix(0, 1, (char*)&val);
    val++;
    cr.setValueInRoughMatrix(1, 0, (char*)&val);
    val++;
    cr.setValueInRoughMatrix(1, 1, (char*)&val);

    int16_t val2 = 1;
    cr.setValueInIntegrateMatrix(0, 0, (char*)&val2);
    val2++;
    cr.setValueInIntegrateMatrix(0, 1, (char*)&val2);
    val2++;
    cr.setValueInIntegrateMatrix(1, 0, (char*)&val2);
    val2++;
    cr.setValueInIntegrateMatrix(1, 1, (char*)&val2);

    string rss, iss;
    cr.dumpRoughMatrix(rss);
    cr.dumpIntegrateMatrix(iss);
    //ofstream ss("/tmp/temp2", ofstream::binary);
    //ss.write(iss.c_str(), iss.size());

    CentroidResource cr2;
    ASSERT_TRUE(cr2.init((void*)rss.c_str(), rss.size(), (void*)iss.c_str(), iss.size()));
    ASSERT_EQ(cr2._roughMatrixSize, (size_t)((2+2*2)*4));
    ASSERT_EQ(cr2._integrateMatrixSize, (size_t)(2*2*2));
}
