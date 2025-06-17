#include <fstream>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "index/centroid_resource.h"
#include "index/centroid_quantizer.h"
#include "framework/index_framework.h"
#undef protected
#undef private

using namespace std;
using namespace mercury;

class CentroidQuantizerTest : public testing::Test
{
public:
    void SetUp()
    {
    }
    
    void TearDown()
    {
    }
};

TEST_F(CentroidQuantizerTest, TestDump)
{
    CentroidResource::Pointer cr(new CentroidResource);
    CentroidResource::RoughMeta rMeta(4, 2, {2, 2});
    CentroidResource::IntegrateMeta iMeta(2, 2, 2);
    ASSERT_TRUE(cr->create(rMeta, iMeta));
    ASSERT_EQ(cr->_roughMatrixSize, size_t((2+2*2)*4));
    ASSERT_EQ(cr->_integrateMatrixSize, size_t(2*2*2));

    int32_t val = 1;
    cr->setValueInRoughMatrix(0, 0, (char*)&val);
    val++;
    cr->setValueInRoughMatrix(0, 1, (char*)&val);
    val++;
    cr->setValueInRoughMatrix(1, 0, (char*)&val);
    val++;
    cr->setValueInRoughMatrix(1, 1, (char*)&val);

    int16_t val2 = 1;
    cr->setValueInIntegrateMatrix(0, 0, (char*)&val2);
    val2++;
    cr->setValueInIntegrateMatrix(0, 1, (char*)&val2);
    val2++;
    cr->setValueInIntegrateMatrix(1, 0, (char*)&val2);
    val2++;
    cr->setValueInIntegrateMatrix(1, 1, (char*)&val2);

    string rss, iss;
    cr->dumpRoughMatrix(rss);
    cr->dumpIntegrateMatrix(iss);

    CentroidResource cr2;
    ASSERT_TRUE(cr2.init((void*)rss.c_str(), rss.size(), (void*)iss.c_str(), iss.size()));
    ASSERT_EQ(cr2._roughMatrixSize, (size_t)((2+2*2)*4));
    ASSERT_EQ(cr2._integrateMatrixSize, (size_t)(2*2*2));
    
    CentroidQuantizer cq;
    cq.set_centroid_resource(cr);

    IndexPackage packageHelper;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    const string file_name = "centroid_quantizer.index";

    ASSERT_TRUE(storage != nullptr);
    ASSERT_TRUE(cq.DumpLevelOneQuantizer(packageHelper));
    ASSERT_TRUE(packageHelper.dump(file_name, storage, false));
}

TEST_F(CentroidQuantizerTest, TestLoad)
{
    IndexPackage packageHelper;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto handlerPtr = storage->open("centroid_quantizer.index", false);

    ASSERT_TRUE(packageHelper.load(handlerPtr, false));
    CentroidQuantizer cq;
    ASSERT_TRUE(cq.LoadLevelOneQuantizer(packageHelper));

    CentroidResource* cr2 = cq.get_centroid_resource();
    ASSERT_EQ(cr2->_roughMatrixSize, (size_t)((2+2*2)*4));
    ASSERT_EQ(cr2->_integrateMatrixSize, (size_t)(2*2*2));
}

TEST_F(CentroidQuantizerTest, TestCreate)
{   
    IndexPackage packageHelper;
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto handlerPtr = storage->open("centroid_quantizer.index", false);

    CentroidQuantizer cq;
    ASSERT_TRUE(packageHelper.load(handlerPtr, false));
    ASSERT_TRUE(cq.LoadLevelOneQuantizer(packageHelper));

    map<string, size_t> stab;
    ASSERT_TRUE(cq.CreateLevelOneQuantizer(stab));
}
