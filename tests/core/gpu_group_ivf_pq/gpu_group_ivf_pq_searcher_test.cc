/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: anduo <anduo@xiaohongshu.com>
/// Created: 2023-01-17 14:40

#ifdef ENABLE_GPU_IN_MERCURY_

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#define protected public
#define private public
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/gpu_group_ivf_pq/gpu_group_ivf_pq_searcher.h"
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_merger.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "tests/core/group_ivf/common.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GpuGroupIvfPqSearcherTest : public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf_pq/test_data/");
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 256);
        index_params_.set(PARAM_INDEX_TYPE, "GroupIvfPq");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        index_params_.set(PARAM_VECTOR_ENABLE_GPU, true);
        index_params_.set(PARAM_VECTOR_INDEX_NAME, "GpuGroupIvfPqSearcherTest");
        index_params_.set(PARAM_VECTOR_ENABLE_DEVICE_NO, 0);
        factory_.SetIndexParams(index_params_);
    }

    void TearDown() {}

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GpuGroupIvfPqSearcherTest, TestSearch)
{
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex *core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex *>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data =
        core_index1->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex *core_index2 = dynamic_cast<mercury::core::GroupIvfPqIndex *>(index2.get());

    SlotIndex label2 = 0;
    const void *data2 =
        core_index2->centroid_resource_manager_.GetCentroidResource(group_index).getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "0:0;1:11;2:200||" + data2_str;
    size_t size2 = core_index2->index_meta_._element_size;
    docid_t docid2 = 0;
    ret = core_index2->Add(docid2, INVALID_PK, group_str);
    ret = core_index2->Add(docid2 + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index2->GetDocNum());

    MergeIdMap id_map;
    id_map.push_back(std::make_pair(1, 1));
    id_map.push_back(std::make_pair(0, 0));
    id_map.push_back(std::make_pair(0, 1));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();

    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);

    GroupIvfPqMerger *ivf_pq_merger = dynamic_cast<GroupIvfPqMerger *>(index_merger.get());
    mercury::core::GroupIvfPqIndex &merged_index = ivf_pq_merger->merged_index_;

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void *dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context1(index_params);
    // normal
    std::string search_str = "100&0:0#0||" + data_str;
    QueryInfo query_info_1(search_str);
    ASSERT_TRUE(query_info_1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info_1, &context1);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context1.Result().size());
    ASSERT_EQ(0, context1.Result().at(0).gloid);
    ASSERT_EQ(1, context1.Result().at(1).gloid);
    ASSERT_EQ(2, context1.Result().at(2).gloid);

    // multi query
    mercury::core::GeneralSearchContext context2(index_params);
    auto pool = std::make_shared<putil::mem_pool::Pool>();
    context2.SetSessionPool(pool.get());
    search_str = "6&0:0#3;0:0#3||" + data_str + ";" + data_str + "||mercury.general.multi_query_mode=true";
    QueryInfo query_info_2(search_str);
    ASSERT_TRUE(query_info_2.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info_2, &context2);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context2.Result().size());
    ASSERT_EQ(0, context2.Result().at(0).gloid);
    ASSERT_EQ(1, context2.Result().at(1).gloid);
    ASSERT_EQ(2, context2.Result().at(2).gloid);
}

MERCURY_NAMESPACE_END(core);

#endif