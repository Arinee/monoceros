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
#include "src/engine/redindex/redindex_index.h"
#include "src/engine/redindex/redindex_factory.h"
#include "src/core/algorithm/ivf_pq/ivf_pq_index.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/utils/string_util.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class RedIndexIvfPQTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(RedIndexIvfPQTest, TestInit)
{
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    SchemaParams schema = { {"DataType", "float"},
        {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQ"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
                            , {"IvfCoarseScanRatio", "0.5"}
                            , {"IvfPQScanNum", "100"}};

    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfPqIndex* core_index = dynamic_cast<mercury::core::IvfPqIndex*>(index->core_index_.get());

    EXPECT_EQ(256, core_index->index_meta_.dimension());
    EXPECT_EQ(IndexMeta::kTypeFloat, core_index->index_meta_.type());
    EXPECT_EQ(IndexDistance::kMethodFloatSquaredEuclidean, core_index->index_meta_.method());
    EXPECT_EQ(14900000, core_index->GetMaxDocNum());
    size_t memQuota = (1024L-100) * 1024L * 1024L;
    size_t elemSize = 256 * sizeof(float);
    std::cout << "---------- test memQuota: " << memQuota << std::endl;
    std::cout << "---------- test elemSize: " << elemSize << std::endl;
    EXPECT_EQ(13400000, core_index->MemQuota2DocCount(memQuota, elemSize));
    EXPECT_FLOAT_EQ(0.5, core_index->index_params_.getFloat(PARAM_COARSE_SCAN_RATIO));
    EXPECT_EQ(2, core_index->GetCentroidResource().getLeafCentroidNum());
    EXPECT_EQ(core_index->GetCoarseIndex().getHeader()->capacity,
              CoarseIndex<BigBlock>::calcSize(2, core_index->GetMaxDocNum()));
    EXPECT_EQ(core_index->coarse_base_.size(),
              CoarseIndex<BigBlock>::calcSize(2, core_index->GetMaxDocNum()));
    EXPECT_EQ(core_index->slot_index_profile_.getHeader()->capacity,
              ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(SlotIndex)));
    EXPECT_EQ(core_index->slot_index_profile_base_.size(),
              ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(SlotIndex)));

    EXPECT_EQ(100, core_index->index_params_.getFloat(PARAM_PQ_SCAN_NUM));
    auto product_size = core_index->GetCentroidResource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    EXPECT_EQ(core_index->pq_code_profile_.getHeader()->capacity,
              ArrayProfile::CalcSize(core_index->GetMaxDocNum(), product_size));
    EXPECT_EQ(core_index->pq_code_base_.size(), ArrayProfile::CalcSize(core_index->GetMaxDocNum(), product_size));
    EXPECT_EQ("", core_index->integrate_matrix_str_);
    EXPECT_EQ(false, core_index->GetCentroidResource()._roughOnly);
    // centroid_num * fragmentsize * elementsize
    EXPECT_EQ(256 * 32 * 8 * 4, core_index->GetCentroidResource()._integrateMatrixSize);
    const void *data = core_index->centroid_resource_.getValueInIntegrateMatrix(0, 0);
    EXPECT_FLOAT_EQ(-0.0578744, ((const float*)data)[0]);
    EXPECT_FLOAT_EQ(-0.0904761, ((const float*)data)[7]);
    const void *data1 = core_index->centroid_resource_.getValueInIntegrateMatrix(31, 255);
    EXPECT_FLOAT_EQ(0.0052324, ((const float*)data1)[1]);
    EXPECT_FLOAT_EQ(-0.0438240, ((const float*)data1)[3]);
}


TEST_F(RedIndexIvfPQTest, TestGetNearestLabel)
{
    SchemaParams schema = { {"DataType", "float"},
                            {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
                            , {"IvfCoarseScanRatio", "0.5"}
                            , {"IvfPQScanNum", "100"}};

    RedIndex::Pointer index = RedIndexFactory::Create(schema);
    mercury::core::IvfPqIndex* core_index = dynamic_cast<mercury::core::IvfPqIndex*>(index->core_index_.get());

    const void *data = core_index->centroid_resource_.getValueInRoughMatrix(0, 1);
    ASSERT_TRUE(data != nullptr);
    EXPECT_EQ(1, core_index->GetNearestLabel(data, core_index->index_meta_.sizeofElement()));
}

TEST_F(RedIndexIvfPQTest, TestAdd)
{
    SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQFlat"},
                          {"TrainDataPath", "tests/engine/redindex/test_data/"}
                          , {"IvfCoarseScanRatio", "0.5"}
                          , {"IvfPQScanNum", "100"}};
    RedIndex::Pointer pindex = RedIndexFactory::Create(schema);
    mercury::core::IvfPqIndex* index = dynamic_cast<mercury::core::IvfPqIndex*>(pindex->core_index_.get());

    SlotIndex label = 1;
    //const void *data = index.centroid_resource_.getValueInRoughMatrix(0, label);
    float data[256] = {0};
    float data1[8] = {-0.0045091,-0.0033348,0.0567824,-0.1140584,-0.0392873,-0.0592759,0.0326772,-0.0554174}; //from integrate_0.dat 52
    float data2[8] = {-0.0813318,0.0767107,-0.1274698,-0.0550630,-0.0254705,-0.0510723,0.0189815,-0.0979695}; //from integrate_31.dat 20
    memcpy(data, data1, 8 * sizeof(float));
    memcpy(&data[248], data2, 8 * sizeof(float));

    std::vector<float> float_vec;
    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);

    size_t size = index->index_meta_.sizeofElement();
    RedIndexDocid redindex_docid = 0;
    int ret = pindex->Add(redindex_docid, data_str);
    ASSERT_EQ(0, ret);
    EXPECT_EQ(label, index->GetNearestLabel(data, size));
    EXPECT_EQ(0, index->coarse_index_.search(label).next());
    EXPECT_EQ(label, *(SlotIndex*)index->slot_index_profile_.getInfo(redindex_docid));

    //auto product_size = index.get_centroid_resource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    const uint16_t* profile = (const uint16_t *)index->GetPqCodeProfile().getInfo(redindex_docid);
    EXPECT_EQ(52, profile[0]);
    EXPECT_EQ(20, profile[31]);
}

TEST_F(RedIndexIvfPQTest, TestDumpAndLoad)
{
    SchemaParams schema = { {"DataType", "float"},
                            {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
                            , {"IvfCoarseScanRatio", "0.5"}
                            , {"IvfPQScanNum", "100"}};

    RedIndex::Pointer pindex = RedIndexFactory::Create(schema);
    mercury::core::IvfPqIndex* index = dynamic_cast<mercury::core::IvfPqIndex*>(pindex->core_index_.get());

    SlotIndex label = 1;
    //const void *data = index.centroid_resource_.getValueInRoughMatrix(0, label);
    float data[256] = {0};
    float data1[8] = {-0.0045091,-0.0033348,0.0567824,-0.1140584,-0.0392873,-0.0592759,0.0326772,-0.0554174}; //from integrate_0.dat 52
    float data2[8] = {-0.0813318,0.0767107,-0.1274698,-0.0550630,-0.0254705,-0.0510723,0.0189815,-0.0979695}; //from integrate_31.dat 20
    memcpy(data, data1, 8 * sizeof(float));
    memcpy(&data[248], data2, 8 * sizeof(float));

    std::vector<float> float_vec;
    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);

    size_t size = index->index_meta_.sizeofElement();
    RedIndexDocid redindex_docid = 0;
    int ret = pindex->Add(redindex_docid, data_str);
    ASSERT_EQ(0, ret);
    const void *dump_data = nullptr;
    size_t dump_size = 0;
    ret = index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    RedIndex::Pointer loaded_index = RedIndexFactory::Load(schema, dump_data, dump_size);
    index = dynamic_cast<mercury::core::IvfPqIndex*>(loaded_index->core_index_.get());

    EXPECT_EQ(label, index->GetNearestLabel(data, size));
    EXPECT_EQ(0, index->coarse_index_.search(label).next());
    EXPECT_EQ(label,
              *(SlotIndex*)index->slot_index_profile_.getInfo(redindex_docid));
    EXPECT_EQ(1, index->GetDocNum());

    //auto product_size = index.get_centroid_resource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    const uint16_t* profile = (const uint16_t *)index->GetPqCodeProfile().getInfo(redindex_docid);
    EXPECT_EQ(52, profile[0]);
    EXPECT_EQ(20, profile[31]);

}

/*
TEST_F(RedIndexIvfPQTest, TestLoadLocal) {
    SchemaParams schema = { {"DataType", "float"},
                            {"Method","L2"}, {"Dimension", "256"}, {"IndexType", "IvfPQFlat"},
                            {"TrainDataPath", "tests/engine/redindex/test_data/"}
                            , {"IvfCoarseScanRatio", "0.5"}
                            , {"IvfPQScanNum", "100"}};

    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/mercury_index.package", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;

    RedIndex::Pointer loaded_index = RedIndexFactory::Load(schema, data, handler->size());
    mercury::core::IvfPqIndex* index = dynamic_cast<mercury::core::IvfPqIndex*>(loaded_index->core_index_.get());

    //EXPECT_EQ(label, index->GetNearestLabel(data, size));
    //EXPECT_EQ(0, index->coarse_index_.search(label).next());
    //EXPECT_EQ(label,
    // *(SlotIndex*)index->slot_index_profile_.getInfo(redindex_docid));
    EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->slotNum);
    EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->capacity);
    EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->usedDocNum);

    EXPECT_EQ(1, index->GetDocNum());

    //auto product_size = index.get_centroid_resource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    const uint16_t* profile = (const uint16_t *)index->GetPqCodeProfile().getInfo(1);
    EXPECT_EQ(52, profile[0]);
    EXPECT_EQ(20, profile[31]);
}
*/
}; // namespace redindex
}; // namespace mercury
