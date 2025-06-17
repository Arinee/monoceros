/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-23 10:59

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>

#define protected public
#define private public
// #include "src/core/framework/index_storage.h"
// #include "src/core/framework/instance_factory.h"
// #include "src/core/utils/string_util.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/group_ivf/group_ivf_index.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfIndexTest : public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 0.5);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 256);
        index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        factory_.SetIndexParams(index_params_);
    }

    void TearDown() {}

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfIndexTest, TestInit)
{
    std::cout << "size of group ivf:" << sizeof(GroupIvfIndex) << std::endl;
    // EXPECT_TRUE(false);
    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index.get());

    ASSERT_TRUE(core_index != nullptr);

    EXPECT_EQ(256, core_index->index_meta_.dimension());
    EXPECT_EQ(IndexMeta::kTypeFloat, core_index->index_meta_.type());
    EXPECT_EQ(IndexDistance::kMethodFloatSquaredEuclidean, core_index->index_meta_.method());
    EXPECT_EQ(1040000, core_index->GetMaxDocNum());
    size_t memQuota = (1024L - 100) * 1024L * 1024L;
    size_t elemSize = 256 * sizeof(float);
    std::cout << "---------- test memQuota: " << memQuota << std::endl;
    std::cout << "---------- test elemSize: " << elemSize << std::endl;
    EXPECT_EQ(940000, core_index->MemQuota2DocCount(memQuota, elemSize));
    memQuota = 100000000;
    EXPECT_EQ(90000, core_index->MemQuota2DocCount(memQuota, 128 * sizeof(float)));
    EXPECT_FLOAT_EQ(0.5, core_index->index_params_.getFloat(PARAM_COARSE_SCAN_RATIO));
    EXPECT_EQ(10, core_index->centroid_resource_manager_.GetTotalCentroidsNum());

    EXPECT_EQ(core_index->GetCoarseIndex().getHeader()->capacity,
              CoarseIndex<SmallBlock>::calcSize(10, core_index->GetMaxCoarseDocNum()));
    EXPECT_EQ(core_index->coarse_base_.size(), CoarseIndex<SmallBlock>::calcSize(10, core_index->GetMaxCoarseDocNum()));
    EXPECT_EQ(68614240, CoarseIndex<SmallBlock>::calcSize(442988, 300000));
    EXPECT_EQ(core_index->feature_profile_.getHeader()->capacity,
              ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(float) * 256));
    EXPECT_EQ(core_index->feature_profile_base_.size(),
              ArrayProfile::CalcSize(core_index->GetMaxDocNum(), sizeof(float) * 256));

    // centroid_num * fragmentsize * elementsize
    CentroidResource &cr = core_index->centroid_resource_manager_.GetCentroidResource(0);
    EXPECT_EQ(true, cr._roughOnly);

    const void *data = cr.getValueInRoughMatrix(0, 0);
    EXPECT_FLOAT_EQ(0.0216151, ((const float *)data)[0]);
    EXPECT_FLOAT_EQ(-0.0903824, ((const float *)data)[7]);
    EXPECT_FLOAT_EQ(0.0122743, ((const float *)data)[255]);
}

TEST_F(GroupIvfIndexTest, TestMemQuotaNoProfile)
{
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, false);
    factory.SetIndexParams(index_params);

    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    Index::Pointer index = factory.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index.get());

    ASSERT_TRUE(core_index != nullptr);

    EXPECT_EQ(256, core_index->index_meta_.dimension());
    EXPECT_EQ(IndexMeta::kTypeFloat, core_index->index_meta_.type());
    EXPECT_EQ(IndexDistance::kMethodFloatSquaredEuclidean, core_index->index_meta_.method());
    EXPECT_EQ(251650000, core_index->GetMaxDocNum());
    size_t memQuota = (1024L - 100) * 1024L * 1024L;
    size_t elemSize = 256 * sizeof(float);
    std::cout << "---------- test memQuota: " << memQuota << std::endl;
    std::cout << "---------- test elemSize: " << elemSize << std::endl;
    EXPECT_EQ(227080000, core_index->MemQuota2DocCount(memQuota, elemSize));
    memQuota = 5000000;
    EXPECT_EQ(1170000, core_index->MemQuota2DocCount(memQuota, 128 * sizeof(float)));
}

TEST_F(GroupIvfIndexTest, TestGetNearestLabel)
{
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index.get());
    ASSERT_TRUE(core_index != nullptr);

    gindex_t group_index = 1;
    CentroidResource &cr = core_index->centroid_resource_manager_.GetCentroidResource(group_index);
    const void *data = cr.getValueInRoughMatrix(0, 1);
    ASSERT_TRUE(data != nullptr);
    EXPECT_EQ(6, core_index->GetNearestGroupLabel(data, core_index->index_meta_.sizeofElement(), group_index,
                                                  core_index->GetCentroidResourceManager()));
}

TEST_F(GroupIvfIndexTest, TestCoarseIndex)
{
    EXPECT_EQ(15242816, CoarseIndex<SmallBlock>::calcSize(100000, 10000));
}

TEST_F(GroupIvfIndexTest, TestAdd)
{
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index.get());
    ASSERT_TRUE(core_index != nullptr);

    SlotIndex label = 1;
    const void *data = core_index->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, label);
    std::vector<float> float_vec;
    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    std::string invalid_group_str = "1:11;2:1000||" + data_str;

    size_t size = core_index->index_meta_._element_size;
    docid_t docid = 1000;
    int ret = index->Add(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    EXPECT_EQ(label, core_index->GetNearestGroupLabel(data, size, 0, core_index->GetCentroidResourceManager()));
    EXPECT_EQ(3, core_index->coarse_index_.getUsedDocNum());
    EXPECT_EQ(1000, core_index->coarse_index_.search(label).next());

    ret = index->Add(docid, INVALID_PK, invalid_group_str);
    ASSERT_EQ(0, ret);
}

TEST_F(GroupIvfIndexTest, TestDumpAndLoad)
{
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index.get());
    ASSERT_TRUE(core_index != nullptr);

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data =
        core_index->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label);

    std::vector<float> float_vec;

    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));

    std::string data_str = StringUtil::vectorToStr(float_vec);
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    std::string invalid_group_str = "1:11;2:1000||" + data_str;

    size_t size = core_index->index_meta_._element_size;
    docid_t docid = 0;
    int ret = index->Add(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    const void *dump_data = nullptr;
    size_t dump_size = 0;
    ret = core_index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    Index::Pointer loaded_index = factory_.CreateIndex(true);
    loaded_index->Load(dump_data, dump_size);
    core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(loaded_index.get());

    EXPECT_EQ(label,
              core_index->GetNearestGroupLabel(data, size, group_index, core_index->GetCentroidResourceManager()));
    EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
    EXPECT_EQ(1040000, core_index->GetMaxDocNum());
    EXPECT_EQ(1, core_index->GetDocNum());
}
/*
TEST_F(GroupIvfIndexTest, TestLoadLocal) {
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 128);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    factory.SetIndexParams(index_params);

    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/group_index/data_rel/package.0", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;

    Index::Pointer loaded_index = factory.CreateIndex(true);
    loaded_index->Load(data, handler->size());
    mercury::core::GroupIvfIndex* index = dynamic_cast<mercury::core::GroupIvfIndex*>(loaded_index.get());
    std::cout << "doc num:" << index->GetMaxDocNum() << std::endl;

    //EXPECT_EQ(label, index->GetNearestLabel(data, size));
    //EXPECT_EQ(0, index->coarse_index_.search(label).next());
    //EXPECT_EQ(label,
    // *(SlotIndex*)index->slot_index_profile_.getInfo(redindex_docid));
    //EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->slotNum);
    //EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->capacity);
    //EXPECT_EQ(1, index->GetCoarseIndex()._pHeader->usedDocNum);


    EXPECT_EQ(1, index->GetMaxDocNum());
    EXPECT_EQ(1, index->GetCoarseIndex().getUsedDocNum());
    EXPECT_EQ(1, index->GetGroupManager().GetGroupNum());

    //auto product_size = index.get_centroid_resource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    }*/

MERCURY_NAMESPACE_END(core);
