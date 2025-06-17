/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_searcher.h"
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_merger.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "../group_ivf/common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfPqSearcherTest: public testing::Test
{
public:
    void SetUp()
        {
            std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
            index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf_pq/test_data/");
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_METHOD, "L2");
            index_params_.set(PARAM_DIMENSION, 256);
            index_params_.set(PARAM_INDEX_TYPE, "GroupIvfPq");
            index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
            index_params_.set(PARAM_PQ_SCAN_NUM, 5000);
            factory_.SetIndexParams(index_params_);
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfPqSearcherTest, TestSearch) {
    std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    std::string data_str = DataToStr(data);
    std::string group_str = data_str;
    size_t size = core_index1->index_meta_._element_size;

    int ret = core_index1->Add(0, INVALID_PK, group_str);
    ret = core_index1->Add(1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());
    LOG_INFO("add index1 success");

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index2.get());
    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    std::string data2_str = DataToStr(data2);
    group_str = "1:11;2:200||" + data2_str;
    size_t size2 = core_index2->index_meta_._element_size;
    ret = core_index2->Add(0, INVALID_PK, group_str);
    ret = core_index2->Add(1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index2->GetDocNum());
    LOG_INFO("add index2 success");

    MergeIdMap id_map;
    id_map.push_back(std::make_pair(1, 1 /*pk*/)); //newid: 0
    id_map.push_back(std::make_pair(0, 0)); //newid: 1
    id_map.push_back(std::make_pair(0, 1)); //newid: 2  //--> 0:2 1:1 2:1 

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    GroupIvfPqMerger* ivf_merger = dynamic_cast<GroupIvfPqMerger*>(index_merger.get());
    mercury::core::GroupIvfPqIndex& merged_index = ivf_merger->merged_index_;
    ASSERT_EQ(3, merged_index.GetDocNum());
    ASSERT_EQ(*(SlotIndex*)merged_index.GetRankScoreProfile().getInfo(0), 0);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    
    std::string search_str = "100&0:0#100||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(1, context.Result().at(0).gloid);
    ASSERT_EQ(2, context.Result().at(1).gloid);
    // ASSERT_EQ(2, context.Result().at(2).gloid);
    LOG_INFO("group ivf pq search success");

    mercury::core::GeneralSearchContext context2(index_params);
    search_str = "100&0:0#5;1:11#5;2:200#5||" + data_str;
    QueryInfo query_info2(search_str);
    ASSERT_TRUE(query_info2.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info2, &context2);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context2.Result().size());
    ASSERT_EQ(0, context2.Result().at(0).gloid);
    ASSERT_EQ(1, context2.Result().at(1).gloid);
    ASSERT_EQ(2, context2.Result().at(2).gloid);
    LOG_INFO("group ivf pq search with both zero and other group info success");

    mercury::core::GeneralSearchContext context3(index_params);
    search_str = "100&1:11#0||" + data_str;
    QueryInfo query_info3(search_str);
    ASSERT_TRUE(query_info3.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info3, &context3);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(1, context3.Result().size());
    LOG_INFO("group ivf search success");

    mercury::core::GeneralSearchContext context4(index_params);
    search_str = "0:0#100||" + data_str + "||mercury.coarse_scan_ratio=0.2,k2=v2"; //0.2*5=1->只找一个中心点
    QueryInfo query_info4(search_str);
    ASSERT_TRUE(query_info4.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info4, &context4);
    ASSERT_EQ(context4.Result().size(), 2);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(1, context4.Result().at(0).gloid);
    ASSERT_EQ(2, context4.Result().at(1).gloid);
    LOG_INFO("group ivf search with params success");

    mercury::core::GeneralSearchContext context5(index_params);
    search_str = "0:0||" + data_str;
    QueryInfo query_info5(search_str);
    ASSERT_FALSE(query_info5.MakeAsSearcher());
    LOG_INFO("group ivf search with no return number success");
}

TEST_F(GroupIvfPqSearcherTest, TestIndexParams) {
    std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());

    IndexParams index_params_ = core_index1->GetIndexParams();
    ASSERT_EQ(32, core_index1->GetCentroidResourceManager().GetCentroidResource(0).getIntegrateMeta().fragmentNum);//-------
    ASSERT_EQ(256, core_index1->GetCentroidResourceManager().GetCentroidResource(0).getIntegrateMeta().centroidNum);//     |
    ASSERT_EQ(32, core_index1->GetCentroidResourceManager().GetCentroidResource(0).getIntegrateMeta().elemSize); //1024 / 32 = 32
    ASSERT_EQ(256*sizeof(float), core_index1->GetIndexMeta().sizeofElement()); //256*4(float) ----------------------|
    ASSERT_EQ(sizeof(uint16_t)*32, core_index1->GetProductSize()); 
    ASSERT_EQ(256, index_params_.getUint64(PARAM_DIMENSION));

    ASSERT_EQ(10, core_index1->GetCentroidResourceManager().GetTotalCentroidsNum());
    ASSERT_EQ(5, core_index1->GetCentroidResourceManager().GetCentroidsNum(0));
    ASSERT_EQ(2, core_index1->GetCentroidResourceManager().GetCentroidsNum(1));
    ASSERT_EQ(2, core_index1->GetCentroidResourceManager().GetCentroidsNum(2));
    ASSERT_EQ(1, core_index1->GetCentroidResourceManager().GetCentroidsNum(3));
}

TEST_F(GroupIvfPqSearcherTest, TestDimensionNotMatch) {
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());

    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0)
        .getValueInRoughMatrix(0, 1);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    int ret = core_index1->Add(0, INVALID_PK, group_str);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    core_index1->index_meta_.setDimension(100);
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "3&1:11#1;2:200#1||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, -1);

    search_str = "100&0:0#1||" + data_str;
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info1, &context);
    ASSERT_EQ(ret_code, -1);
}

TEST_F(GroupIvfPqSearcherTest, TestParallel) {
    putenv("mercury_need_parallel=true");
    //putenv("mercury_concurrency=3");
    //putenv("mercury_no_concurrency_count=1");
    putenv("mercury_pool_size=10");
    putenv("mercury_min_parralel_centroids=1");
    putenv("mercury_doc_num_per_concurrency=100");

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());

    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 100; i++) {
        std::string group_str = "1:11||" + DataToStr(data);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    for (int i = 100; i < 200; i++) {
        std::string group_str = "2:200||" + DataToStr(data1);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    //0:0 -> 200, 1:11 -> 100, 2:200 -> 100

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    int ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "50&0:0#50||" + DataToStr(data);
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(0, context.Result().size());
    LOG_INFO("parallel with single pq group success");

    mercury::core::GeneralSearchContext context1(index_params);
    search_str = "100&1:11#50||" + DataToStr(data);
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info1, &context1);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(100, context1.Result().size());
    LOG_INFO("parallel with single normal group success");

    mercury::core::GeneralSearchContext context2(index_params);
    search_str = "100&2:200#50||" + DataToStr(data);
    QueryInfo query_info2(search_str);
    ASSERT_TRUE(query_info2.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info2, &context2);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(100, context2.Result().size());
    LOG_INFO("parallel with single normal group success");

    mercury::core::GeneralSearchContext context3(index_params);
    search_str = "100&1:11#50;2:200#50||" + DataToStr(data);
    QueryInfo query_info3(search_str);
    ASSERT_TRUE(query_info3.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info3, &context3);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(100, context3.Result().size());
    LOG_INFO("parallel with 2 groups success");

    mercury::core::GeneralSearchContext context4(index_params);
    search_str = "200&0:0#50;1:11#50;2:200#100||" + DataToStr(data);
    QueryInfo query_info4(search_str);
    ASSERT_TRUE(query_info4.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info4, &context4);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(150, context4.Result().size());
    LOG_INFO("parallel with 3 groups success");
    
    clearenv();
}

TEST_F(GroupIvfPqSearcherTest, TestRecallMode) {
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());

    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 100; i++) {
        std::string group_str = "1:11||" + DataToStr(data);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    for (int i = 100; i <= 200; i++) {
        std::string group_str = "2:200||" + DataToStr(data1);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    //0:0 -> 200, 1:11 -> 100, 2:200 -> 100

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    int ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);

    std::string search_str = "3&0:0#1;1:11#1;2:200#1||" + DataToStr(data) + "||mercury.coarse_scan_ratio=1,mercury.general.recall_test_mode=v2";
    QueryInfo query_info5(search_str);
    ASSERT_TRUE(query_info5.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info5, &context);

    ASSERT_EQ(ret_code, 0);
}

TEST_F(GroupIvfPqSearcherTest, TestMultiQuery) {
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index1.get());

    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 100; i++) {
        std::string group_str = "1:11||" + DataToStr(data);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    for (int i = 100; i < 200; i++) {
        std::string group_str = "2:200||" + DataToStr(data1);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    int ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string data_str = DataToStr(data);
    std::string data_str1 = DataToStr(data1);
    std::string search_str = "100&1:11#50;2:200#50||" + data_str + ";" + data_str1 + "||mercury.general.multi_query_mode=true";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(100, context.Result().size());

    mercury::core::GeneralSearchContext context1(index_params);
    search_str = "200&0:0#50;1:11#50;2:200#100||" + data_str + ";" + data_str1 + ";" + data_str1 + "||mercury.general.multi_query_mode=true";
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info1, &context1);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(150, context1.Result().size());
}



MERCURY_NAMESPACE_END(core);
