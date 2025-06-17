/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-23 10:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/core/algorithm/centroid_resource_manager.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class CentroidResourceManagerTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

CentroidResource MockCentroidResource(const std::vector<float>& centroid, size_t centroid_num) {
    CentroidResource centroid_resource;
    std::vector<uint32_t> centroids_levelcnts;
    centroids_levelcnts.push_back(centroid_num);
    CentroidResource::RoughMeta rough_meta(sizeof(float) * 256, DefaultLevelCnt, centroids_levelcnts);
    if (!centroid_resource.create(rough_meta)) {
        std::cerr << "Failed to create centroid resource." << std::endl;
        return CentroidResource();
    }

    //only 1 level
    for (size_t i = 0; i < centroid_num; i++) {
        if (!centroid_resource.setValueInRoughMatrix(0, i, centroid.data())) {
            std::cerr << "Failed to set centroid resource rough matrix." << std::endl;
            return CentroidResource();
        }
    }

    return centroid_resource;
}

TEST_F(CentroidResourceManagerTest, TestSimple)
{
    char* buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    std::vector<float> mock1(256, 0.0);
    CentroidResource cr1 = MockCentroidResource(mock1, 100);
    std::vector<float> mock2(256, 1.1);
    CentroidResource cr2 = MockCentroidResource(mock2, 1);
    CentroidResourceManager crm;
    crm.AddCentroidResource(std::move(cr1));
    crm.AddCentroidResource(std::move(cr2));
    ASSERT_EQ(101, crm.GetTotalCentroidsNum());
    ASSERT_EQ(1, crm.GetCentroidsNum(1));
    ASSERT_EQ(15, crm.GetSlotIndex(0, 15));
    ASSERT_EQ(100, crm.GetSlotIndex(1, 0));
}

TEST_F(CentroidResourceManagerTest, TestDumpLoad)
{
    std::vector<float> mock1(256, 0.0);
    CentroidResource cr1 = MockCentroidResource(mock1, 100);
    std::vector<float> mock2(256, 1.1);
    CentroidResource cr2 = MockCentroidResource(mock2, 1);
    CentroidResourceManager crm;
    crm.AddCentroidResource(std::move(cr1));
    crm.AddCentroidResource(std::move(cr2));

    std::string rough;
    crm.DumpRoughMatrix(rough);
    CentroidResourceManager crm2;
    crm2.LoadRough(rough.data(), rough.size());
    ASSERT_EQ(101, crm2.GetTotalCentroidsNum());
    ASSERT_EQ(1, crm2.GetCentroidsNum(1));
    ASSERT_EQ(15, crm2.GetSlotIndex(0, 15));
    ASSERT_EQ(100, crm2.GetSlotIndex(1, 0));
}

MERCURY_NAMESPACE_END(core);
