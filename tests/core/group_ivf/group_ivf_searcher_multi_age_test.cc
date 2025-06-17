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
#include "src/core/algorithm/group_ivf/group_ivf_searcher.h"
#include "src/core/algorithm/group_ivf/group_ivf_merger.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/algorithm/thread_common.h"
#include "common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/algorithm/group_ivf/mocked_vector_reader.h"
#include "src/core/utils/note_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfSearcherMultiAgeTest: public testing::Test
{
public:
    void SetUp()
        {
            char* buffer;
            buffer = getcwd(NULL, 0);
            std::cout << "cwd is:" << buffer << std::endl;
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
            index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_METHOD, "L2");
            index_params_.set(PARAM_DIMENSION, 256);
            index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
            index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
            index_params_.set(PARAM_MULTI_AGE_MODE, true);
            factory_.SetIndexParams(index_params_);
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfSearcherMultiAgeTest, TestNormSearchInMultiAgeIndex) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 0;
    const void *data = core_index->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "0:0||" + data_str;
    
    docid_t docid = 0;
    // add a latest doc
    int64_t now_timestamp_s = butil::gettimeofday_s();
    char hexStr[9];
    snprintf(hexStr, 9, "%08lx", now_timestamp_s);
    std::string mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";

    int ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(0.5hour < note age < 0)
    int64_t in_half_hour_timestamp_s = now_timestamp_s - 900;
    snprintf(hexStr, 9, "%08lx", in_half_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(1hour < note age < 0.5hour)
    int64_t in_1_hour_timestamp_s = now_timestamp_s - 2700;
    snprintf(hexStr, 9, "%08lx", in_1_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(2hour < note age < 1hour)
    int64_t in_2_hour_timestamp_s = now_timestamp_s - 5400;
    snprintf(hexStr, 9, "%08lx", in_2_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);
    
    ASSERT_EQ(4, core_index->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    ASSERT_TRUE(dump_data != nullptr);
    ret = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    std::string search_str = "0:0#100||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(4, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
    ASSERT_EQ(3, context.Result().at(3).gloid);
}

TEST_F(GroupIvfSearcherMultiAgeTest, TestMultiAgeSearchInMultiAgeIndex) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 0;
    const void *data = core_index->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "0:0||" + data_str;
    
    docid_t docid = 0;
    // add a latest doc
    int64_t now_timestamp_s = butil::gettimeofday_s();
    char hexStr[9];
    snprintf(hexStr, 9, "%08lx", now_timestamp_s);
    std::string mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";

    int ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(0.5hour < note age < 0)
    int64_t in_half_hour_timestamp_s = now_timestamp_s - 900;
    snprintf(hexStr, 9, "%08lx", in_half_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(1hour < note age < 0.5hour)
    int64_t in_1_hour_timestamp_s = now_timestamp_s - 2700;
    snprintf(hexStr, 9, "%08lx", in_1_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);

    // add a doc(2hour < note age < 1hour)
    int64_t in_2_hour_timestamp_s = now_timestamp_s - 5300;
    snprintf(hexStr, 9, "%08lx", in_2_hour_timestamp_s);
    mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
        
    ret = core_index->Add(docid++, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);
    
    ASSERT_EQ(4, core_index->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    ASSERT_TRUE(dump_data != nullptr);
    ret = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    auto pool = std::make_shared<putil::mem_pool::Pool>();
    context.SetSessionPool(pool.get());

    // test all recall
    std::string search_str = "0:0#100||" + data_str + "||" + PARAM_MULTI_AGE_MODE + "=1800#10;3600#10;7200#10";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(4, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
    ASSERT_EQ(3, context.Result().at(3).gloid);

    context.clean();

    // test recall note in 1800 3600, 3600 will extend to 5400 in query info
    search_str = "0:0#100||" + data_str + "||" + PARAM_MULTI_AGE_MODE + "=1800#10;3600#10";
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret = index_searcher->Search(query_info1, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(4, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
    ASSERT_EQ(3, context.Result().at(3).gloid);

    context.clean();

    // test recall note in 1800 7200
    search_str = "0:0#100||" + data_str + "||" + PARAM_MULTI_AGE_MODE + "=1800#10;7200#10";
    QueryInfo query_info2(search_str);
    ASSERT_TRUE(query_info2.MakeAsSearcher());
    ret = index_searcher->Search(query_info2, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(4, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
    ASSERT_EQ(3, context.Result().at(3).gloid);

    context.clean();

    // test recall note exclude 1800-3600, 3600 will extend to 5400 in query info
    search_str = "0:0#100||" + data_str + "||" + PARAM_MULTI_AGE_MODE + "=1800#10; 3600#0;7200#10";
    QueryInfo query_info3(search_str);
    ASSERT_TRUE(query_info3.MakeAsSearcher());
    ret = index_searcher->Search(query_info3, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);

    context.clean();

    // test recall note exclude 0-1800
    search_str = "0:0#100||" + data_str + "||" + PARAM_MULTI_AGE_MODE + "=1800#0; 3600#10;7200#10";
    QueryInfo query_info4(search_str);
    ASSERT_TRUE(query_info4.MakeAsSearcher());
    ret = index_searcher->Search(query_info4, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(2, context.Result().at(0).gloid);
    ASSERT_EQ(3, context.Result().at(1).gloid);
    // ASSERT_EQ(3, context.Result().at(3).gloid);
}

MERCURY_NAMESPACE_END(core);
