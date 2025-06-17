/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/engine/redindex/redindex_index.h"
#include "src/engine/redindex/redindex_factory.h"
#include "src/core/algorithm/ivf/ivf_index.h"
#include "src/core/utils/string_util.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;


class RedIndexIvfTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(RedIndexIvfTest, TestInit)
{
    std::cout<<"sizeof centroid resource:"<<sizeof(mercury::core::CentroidResource)<<std::endl;
    SchemaParams schema = { {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
                            , {"IvfCoarseScanRatio", "0.5"}};

    //std::ifstream sIn("test_data/rough_matrix");
    //ASSERT_TRUE(!!sIn == true);
    //schema.at("IvfCentroidData") = static_cast<std::stringstream const&>(std::stringstream() << sIn.rdbuf()).str();

    RedIndex::Pointer index_p = RedIndexFactory::Create(schema);
    RedIndex& index = *index_p.get();
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index.core_index_.get());

    EXPECT_EQ(256, core_index->index_meta_.dimension());
    EXPECT_EQ(mercury::core::IndexMeta::kTypeFloat, core_index->index_meta_.type());
    EXPECT_EQ(mercury::core::IndexDistance::kMethodFloatSquaredEuclidean, core_index->index_meta_.method());
    EXPECT_EQ(134000000, core_index->GetMaxDocNum());
    size_t memQuota = (1024L-100) * 1024L * 1024L;
    size_t elemSize = 256 * sizeof(float);
    std::cout << "---------- test memQuota: " << memQuota << std::endl;
    std::cout << "---------- test elemSize: " << elemSize << std::endl;
    EXPECT_EQ(120900000, core_index->MemQuota2DocCount(memQuota, elemSize));
    EXPECT_FLOAT_EQ(0.5, core_index->index_params_.getFloat(PARAM_COARSE_SCAN_RATIO));
    EXPECT_EQ(2, core_index->centroid_resource_.getLeafCentroidNum());
    EXPECT_EQ(core_index->coarse_index_.getHeader()->capacity,
              mercury::core::CoarseIndex<BigBlock>::calcSize(2, core_index->GetMaxDocNum()));
    EXPECT_EQ(core_index->coarse_base_.size(),
              mercury::core::CoarseIndex<BigBlock>::calcSize(2, core_index->GetMaxDocNum()));
    EXPECT_EQ(core_index->slot_index_profile_.getHeader()->capacity,
              mercury::core::ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(SlotIndex)));
    EXPECT_EQ(core_index->slot_index_profile_base_.size(),
              mercury::core::ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(SlotIndex)));
}

TEST_F(RedIndexIvfTest, TestGetNearestLabel)
{
    SchemaParams schema = { {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
        , {"IvfCoarseScanRatio", "0.5"}};

    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index->core_index_.get());

    const void *data = core_index->centroid_resource_.getValueInRoughMatrix(0, 1);
    ASSERT_TRUE(data != nullptr);
    EXPECT_EQ(1, core_index->GetNearestLabel(data, core_index->index_meta_._element_size));
}

TEST_F(RedIndexIvfTest, TestAdd)
{
    SchemaParams schema = { {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"},
                            {"IvfCoarseScanRatio", "0.5"}};

    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index->core_index_.get());

    SlotIndex label = 1;
    const void *data = core_index->centroid_resource_.getValueInRoughMatrix(0, label);
    std::vector<float> float_vec;

    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);

    size_t size = core_index->index_meta_._element_size;
    RedIndexDocid redindex_docid = 1000;
    int ret = index->Add(redindex_docid, data_str);
    ASSERT_EQ(0, ret);
    EXPECT_EQ(label, core_index->GetNearestLabel(data, size));
    EXPECT_EQ(1000, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(label, *(SlotIndex*)core_index->slot_index_profile_.getInfo(redindex_docid));
}

TEST_F(RedIndexIvfTest, TestDumpAndLoad)
{
    SchemaParams schema = { {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"},
        {"IvfCoarseScanRatio", "0.5"}};

    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index->core_index_.get());

    SlotIndex label = 1;
    const void *data = core_index->centroid_resource_.getValueInRoughMatrix(0, label);

    std::vector<float> float_vec;

    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);

    size_t size = core_index->index_meta_._element_size;
    RedIndexDocid redindex_docid = 0;
    int ret = index->Add(redindex_docid, data_str);
    ASSERT_EQ(0, ret);
    const void *dump_data = nullptr;
    size_t dump_size = 0;
    ret = core_index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    RedIndex::Pointer loaded_index = RedIndexFactory::Load(schema, dump_data, dump_size);
    core_index = dynamic_cast<mercury::core::IvfIndex*>(loaded_index->core_index_.get());

    EXPECT_EQ(label, core_index->GetNearestLabel(data, size));
    EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(label, 
              *(SlotIndex*)core_index->slot_index_profile_.getInfo(redindex_docid));
    EXPECT_EQ(10001, core_index->GetMaxDocNum());
    EXPECT_EQ(1, core_index->GetDocNum());
}

TEST_F(RedIndexIvfTest, TestMemQuota2DocCount)
{
    size_t memQuota = (1024-100) * 1024L * 1024L;
    size_t elemSize = 256 * sizeof(float);
    size_t elem_count = 116700000;
    size_t slot_num = 8192;

    SchemaParams schema = { {"DataType", "float"},
                            {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfFlat"}};
    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index->core_index_.get());
    core_index->centroid_resource_._roughMeta.levelCnt = 1;
    core_index->centroid_resource_._roughMeta.centroidNums.push_back(slot_num);
    EXPECT_EQ(slot_num, core_index->centroid_resource_.getLeafCentroidNum());
    EXPECT_EQ(elem_count, core_index->MemQuota2DocCount(memQuota, elemSize));
    //EXPECT_EQ(967963264L, mercury::core::CoarseIndex<BigBlock>::calcSize(slot_num, elem_count) +
    //elem_count * sizeof(RedIndexDocid) +
    //ArrayProfile::CalcSize(elem_count, sizeof(SlotIndex)));
}

}; // namespace redindex
}; // namespace mercury
