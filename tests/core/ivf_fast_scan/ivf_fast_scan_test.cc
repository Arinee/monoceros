/// Copyright (c) 2024, xiaohongshu Inc. All rights reserved.
/// Author: haiming <shiyang1@xiaohongshu.com>
/// Created: 2024-09-04 10:45

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/algorithm/ivf_fast_scan/ivf_fast_scan_builder.h"
#include "src/core/algorithm/ivf_fast_scan/ivf_fast_scan_merger.h"
#include "src/core/algorithm/ivf_fast_scan/ivf_fast_scan_searcher.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "../group_ivf/common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class IvfFastScanTest: public testing::Test
{
public:
    void SetUp()
        {
            std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
            index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/ivf_fast_scan/test_data");
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_PQ_FRAGMENT_NUM, 32);
            index_params_.set(PARAM_PQ_CENTROID_NUM, 16);
            index_params_.set(PARAM_METHOD, "L2");
            index_params_.set(PARAM_DIMENSION, 128);
            index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 3000);
            index_params_.set(PARAM_INDEX_TYPE, "IvfFastScan");
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(IvfFastScanTest, TestBuild) {

    index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, false);
    factory_.SetIndexParams(index_params_);

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::IvfFastScanIndex* core_index1 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index1.get());

    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);

    for (int i = 0; i < 50; i++) {
        std::string group_str = "0:0||" + DataToStr(data, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 50; i < 100; i++) {
        std::string group_str = "0:0||" + DataToStr(data1, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }
    size_t dump_size;
    const void* dump_data = nullptr;
    int ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
}

TEST_F(IvfFastScanTest, TestMerge) {

    index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, false);
    factory_.SetIndexParams(index_params_);

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::IvfFastScanIndex* core_index1 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index1.get());

    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);

    for (int i = 0; i < 31; i++) {
        std::string group_str = "0:0||" + DataToStr(data1, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::IvfFastScanIndex* core_index2 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index2.get());

    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 33; i++) {
        std::string group_str = "0:0||" + DataToStr(data2, 128);
        core_index2->Add(i, INVALID_PK, group_str);
    }

    MergeIdMap id_map;
    for (int i = 0; i < 33; i++) {
        id_map.push_back(std::make_pair(1, i));
    }
    for (int i = 0; i < 31; i++) {
        id_map.push_back(std::make_pair(0, i));
    }

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    LOG_INFO("dump size is %lu", dump_size);
    ASSERT_EQ(53568, dump_size);
}

TEST_F(IvfFastScanTest, TestLoad) {

    index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, false);
    factory_.SetIndexParams(index_params_);

    Index::Pointer index1 = factory_.CreateIndex();

    mercury::core::IvfFastScanIndex* core_index1 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index1.get());

    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);

    for (int i = 0; i < 31; i++) {
        std::string group_str = "0:0||" + DataToStr(data1, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::IvfFastScanIndex* core_index2 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index2.get());

    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 33; i++) {
        std::string group_str = "0:0||" + DataToStr(data2, 128);
        core_index2->Add(i, INVALID_PK, group_str);
    }

    MergeIdMap id_map;
    for (int i = 0; i < 33; i++) {
        id_map.push_back(std::make_pair(1, i));
    }
    for (int i = 0; i < 31; i++) {
        id_map.push_back(std::make_pair(0, i));
    }

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    LOG_INFO("dump size is %lu", dump_size);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");
}

TEST_F(IvfFastScanTest, TestSearch) {

    index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, false);
    factory_.SetIndexParams(index_params_);

    Index::Pointer index1 = factory_.CreateIndex();

    mercury::core::IvfFastScanIndex* core_index1 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index1.get());

    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);

    for (int i = 0; i < 999; i++) {
        std::string group_str = "0:0||" + DataToStr(data1, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    Index::Pointer index2 = factory_.CreateIndex();

    mercury::core::IvfFastScanIndex* core_index2 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index2.get());

    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 555; i++) {
        std::string group_str = "0:0||" + DataToStr(data2, 128);
        core_index2->Add(i, INVALID_PK, group_str);
    }

    MergeIdMap id_map;
    for (int i = 0; i < 555; i++) {
        id_map.push_back(std::make_pair(1, i));
    }
    for (int i = 0; i < 999; i++) {
        id_map.push_back(std::make_pair(0, i));
    }

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    LOG_INFO("dump size is %lu", dump_size);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    const void *data3 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 2);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string data_str = DataToStr(data2, 128);
    std::string search_str = "556&0:0#556||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(556, context.Result().size());
    ASSERT_NEAR(context.Result().at(554).score, 0.200304, 0.001f);
    ASSERT_NEAR(context.Result().at(555).score, 1.04254, 0.001f);
}

TEST_F(IvfFastScanTest, TestRecallMode) {

    index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);

    factory_.SetIndexParams(index_params_);

    Index::Pointer index1 = factory_.CreateIndex();

    mercury::core::IvfFastScanIndex* core_index1 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index1.get());

    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);

    for (int i = 0; i < 999; i++) {
        std::string group_str = "0:0||" + DataToStr(data1, 128);
        core_index1->Add(i, INVALID_PK, group_str);
    }

    Index::Pointer index2 = factory_.CreateIndex();

    mercury::core::IvfFastScanIndex* core_index2 = dynamic_cast<mercury::core::IvfFastScanIndex*>(index2.get());

    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);

    for (int i = 0; i < 555; i++) {
        std::string group_str = "0:0||" + DataToStr(data2, 128);
        core_index2->Add(i, INVALID_PK, group_str);
    }

    MergeIdMap id_map;
    for (int i = 0; i < 555; i++) {
        id_map.push_back(std::make_pair(1, i));
    }
    for (int i = 0; i < 999; i++) {
        id_map.push_back(std::make_pair(0, i));
    }

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();
    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);
    LOG_INFO("merge success");

    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    LOG_INFO("dump size is %lu", dump_size);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);
    LOG_INFO("load index success");

    const void *data3 = core_index1->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 2);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string data_str = DataToStr(data2, 128);
    std::string search_str = "556&0:0#556||" + data_str + "||mercury.general.recall_test_mode=true,mercury.coarse_scan_ratio=1.0";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(556, context.Result().size());
    ASSERT_NEAR(context.Result().at(554).score, 0, 0.001f);
    ASSERT_NEAR(context.Result().at(555).score, 0.9, 0.001f);
}

MERCURY_NAMESPACE_END(core);
