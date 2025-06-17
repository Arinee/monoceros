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
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfSearcherTest: public testing::Test
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
            factory_.SetIndexParams(index_params_);
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfSearcherTest, TestSearch) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfIndex*>(index2.get());

    SlotIndex label2 = 0;
    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

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

    GroupIvfMerger* ivf_merger = dynamic_cast<GroupIvfMerger*>(index_merger.get());
    mercury::core::GroupIvfIndex& merged_index = ivf_merger->merged_index_;

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "0:0#100||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
}

TEST_F(GroupIvfSearcherTest, TestContextParams) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11;2:200||" + data_str + "||mercury.coarse_scan_ratio=0.5";
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());


    SlotIndex label2 = 0;
    const void *data2 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "1:11;2:200||" + data2_str;
    ret = core_index1->Add(docid + 2, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 3, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data;
    core_index1->Dump(dump_data, dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "0:0#100||" + data_str + "||mercury.coarse_scan_ratio=0.2,k2=v2";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(context.Result().size(), 0);
    ASSERT_EQ(ret_code, 0);

    // ASSERT_EQ(0, context.Result().at(0).gloid);
    // ASSERT_EQ(1, context.Result().at(1).gloid);
}

TEST_F(GroupIvfSearcherTest, TestVisitLimit) {
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 1);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GROUP_IVF_VISIT_LIMIT, 1);
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    factory.SetIndexParams(index_params);

    Index::Pointer index1 = factory.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());

    SlotIndex label2 = 0;
    const void *data2 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "1:11;2:200||" + data2_str;
    ret = core_index1->Add(docid + 2, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 3, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory.CreateSearcher();
    size_t dump_size;
    const void* dump_data;
    core_index1->Dump(dump_data, dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params1;
    mercury::core::GeneralSearchContext context(index_params1);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "0:0#100||" + data_str + "||mercury.coarse_scan_ratio=1.0";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(0, ret_code);
    ASSERT_EQ(0, context.Result().size());

    // ASSERT_EQ(0, context.Result().at(0).gloid);
    // ASSERT_EQ(1, context.Result().at(1).gloid);
}

TEST_F(GroupIvfSearcherTest, TestRecallMode) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());


    SlotIndex label2 = 0;
    const void *data2 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "2:200||" + data2_str;
    ret = core_index1->Add(docid + 2, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 3, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data;
    core_index1->Dump(dump_data, dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "2&1:11#1;2:200#1||" + data_str + "||mercury.coarse_scan_ratio=1,mercury.general.recall_test_mode=v2";
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(context.Result().size(), 2);
    ASSERT_EQ(ret_code, 0);

    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(2, context.Result().at(1).gloid);
}

TEST_F(GroupIvfSearcherTest, TestSortResult) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label0 = 3;
    const void *data0 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label0);
    std::string data_str0 = DataToStr(data0);
    data_str0[0] = '2';
    std::string group_str0 = "1:11;2:200||" + data_str0;

    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str0);
    ret = core_index1->Add(1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "0:0#3||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(0, context.Result().size());
    // ASSERT_EQ(1, context.Result().at(0).gloid);
    // ASSERT_EQ(2, context.Result().at(1).gloid);
    // ASSERT_EQ(3, context.Result().at(2).gloid);
}

TEST_F(GroupIvfSearcherTest, TestDuplicate) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str);
    ret = core_index1->Add(1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "1:11#10;2:200#5||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(4, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
    ASSERT_EQ(3, context.Result().at(3).gloid);
}

TEST_F(GroupIvfSearcherTest, TestDuplicate2) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str);
    std::string group_str1 = "2:200||" + data_str;
    ret = core_index1->Add(1, INVALID_PK, group_str1);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "1:11#10;2:200#1||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(2, context.Result().at(1).gloid);
}

TEST_F(GroupIvfSearcherTest, TestMultiCentroids) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, 2);
    std::string data_str = DataToStr(data);
    std::string data_str1 = DataToStr(data1);

    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str);

    std::string group_str1 = "2:200||" + data_str1;
    ret = core_index1->Add(1, INVALID_PK, group_str1);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "0:0#1||" + data_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(0, context.Result().size());
    // ASSERT_EQ(0, context.Result().at(0).gloid);
}

TEST_F(GroupIvfSearcherTest, TestTotalRecall) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str);
    std::string group_str1 = "2:200||" + data_str;
    ret = core_index1->Add(1, INVALID_PK, group_str1);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
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

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);

    GeneralSearchContext context1(index_params);
    search_str = "2&1:11#1;2:200#1||" + data_str;
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info1, &context1);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context1.Result().size());
    ASSERT_EQ(0, context1.Result().at(0).gloid);
    ASSERT_EQ(1, context1.Result().at(1).gloid);
}

TEST_F(GroupIvfSearcherTest, TestDimensionNotMatch) {
    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
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
}

TEST_F(GroupIvfSearcherTest, TestParrallel) {
    putenv("mercury_need_parallel=true");
    putenv("mercury_concurrency=2");
    putenv("mercury_no_concurrency_count=1");
    putenv("mercury_pool_size=10");
    putenv("mercury_min_parralel_centroids=1");

    char *buffer;
    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    SlotIndex label1 = 0;
    const void *data1 = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label1);
    std::string data_str = DataToStr(data);
    std::vector<const void*> datas;
    datas.push_back(data);
    datas.push_back(data1);

    std::string data_str1 = DataToStr(data1);
    std::string data_com_str = DataToStr(datas);

    std::string group_str = "1:11||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(0, INVALID_PK, group_str);
    std::string group_str1 = "2:200||" + data_str1;
    ret = core_index1->Add(1, INVALID_PK, group_str1);
    ASSERT_EQ(0, ret);
    std::string group_str2 = "1:11||" + data_str;
    ret = core_index1->Add(2, INVALID_PK, group_str2);
    std::string group_str3 = "2:200||" + data_str;
    ret = core_index1->Add(3, INVALID_PK, group_str3);
    ASSERT_EQ(4, core_index1->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void* dump_data = nullptr;
    ret = core_index1->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    IndexParams index_params;
    GeneralSearchContext context(index_params);
    std::string search_str = "3&1:11#1;2:200#1||" + data_com_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_FLOAT_EQ(0.0, context.Result().at(1).score);

    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(3, context.Result().at(1).gloid);

    GeneralSearchContext context1(index_params);
    search_str = "2&1:11#1;2:200#1||" + data_str;
    QueryInfo query_info1(search_str);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info1, &context1);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context1.Result().size());
    ASSERT_EQ(0, context1.Result().at(0).gloid);
    ASSERT_EQ(3, context1.Result().at(1).gloid);

    // multi query
    GeneralSearchContext context2(index_params);
    search_str = "2&1:11#1;2:200#1||" + data_str + ";" + data_str + "||mercury.general.multi_query_mode=true";
    QueryInfo query_info2(search_str);
    ASSERT_TRUE(query_info2.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info2, &context2);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(2, context2.Result().size());
    ASSERT_EQ(0, context2.Result().at(0).gloid);
    ASSERT_EQ(3, context2.Result().at(1).gloid);

    ASSERT_EQ(mercury_need_parallel, true);
    ASSERT_EQ(mercury_concurrency, 2);

    clearenv();
}

/*
TEST_F(GroupIvfSearcherTest, TestRedIndexGroupSearch) {
    using namespace mercury::tests/engine/redindex;
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    SchemaParams schema = { {"DataType", "float"},
                          {"Method","L2"}, {"Dimension", "128"}, {"IndexType", "GroupIvf"},
                          {"TrainDataPath", "tests/core/group_ivf/test_data/"}
                          , {"IvfCoarseScanRatio", "1"}
                          , {"IvfPQScanNum", "100"}};
  std::string search_str = "2:15#4||0.1937425 0.094472274 0.10508573 0.045744196 -0.25668398 0.0379673 -0.29134563 -0.057198524 0.17343993 -0.1590629 0.045455657 -0.017588628 -0.040428065 -0.14204784 -0.0047223596 -0.042574648 -0.030311871 -0.077819556 0.11680489 0.09542703 -0.05718409 0.10076124 0.210035 0.06057319 0.05447061 -0.007985096 -0.2473974 0.09950372 -0.08264447 -0.08158026 -0.10411242 -0.13155966 -0.08491686 -0.2619516 0.03722058 0.09243979 -0.064065315 0.0034141475 0.022315586 0.13261653 -0.06535373 0.07445833 -0.06791163 8.32867E-4 -0.11108046 0.019320695 -0.1012313 0.15870693 -0.086525135 -0.12584081 -0.2058369 0.21589908 -0.12543581 -0.11908535 0.078234844 -0.12241974 0.05630937 0.06772284 0.25324014 0.0669807 0.028000068 0.19141735 -0.06999357 0.21036644 -0.013573 0.061843 -0.023744 -0.039923 -0.001857 0.256169 -0.041716 -0.039592 0.029918 -0.102778 0.047089 -0.301369 -0.165893 -0.050174 -0.016174 -0.032021 -0.054075 0.163893 -0.103024 0.293106 -0.114674 0.118469 0.054157 0.147814 -0.037121 0.029423 -0.027396 0.107032 -0.133319 0.147285 0.05926 -0.23798 -0.056614 0.073493 0.137862 -0.068934 0.079818 -0.168767 -0.079246 -0.131957 0.083007 0.106564 -0.015183 -0.155893 0.02945 0.07733 -0.004695 -0.147587 -0.105529 0.162875 0.045568 -0.076365 0.203005 -0.049619 0.115688 0.150206 0.138801 0.152891 -0.338256 0.067063 0.167081 -0.075215 -0.046073 -0.5";

  IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(index_storage);
  IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/group_index/mercury_index.package3", false);
  ASSERT_TRUE(handler);
  void *data = nullptr;
  ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

  std::cout<<"handler size: "<<handler->size()<<std::endl;
  IndexSearcherFactory searcher_factory;
  IndexSearcher::Pointer index_searcher = searcher_factory.Create(schema);
  int ret_code = index_searcher->LoadIndex(data, handler->size());
  ASSERT_EQ(ret_code, 0);
  QueryInfo query_info(search_str);
  ASSERT_TRUE(query_info.MakeAsSearcher());
  ret_code = index_searcher->Search(query_info, &context);
  search_iter->Init();
  while (search_iter->IsValid()) {
      docid_t docid = search_iter->Data();
      //IE_LOG(ERROR, "debug:%d", docid);
      if (docid != INVALID_DOC_ID) {
          std::cout<<"docid:"<<docid<<std::endl;
      }
      search_iter->Next();
  }

  ASSERT_EQ(0, search_iter->Data());
  search_iter->Next();
  ASSERT_EQ(1, search_iter->Data());
  search_iter->Next();
  ASSERT_EQ(2, search_iter->Data());
  search_iter->Next();
}
*/
/*
TEST_F(GroupIvfSearcherTest, TestLoadLocal) {
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 1);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "IP");
    index_params.set(PARAM_DIMENSION, 128);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    factory.SetIndexParams(index_params);

    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    //IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/access/ms/mercury_index.package", false);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/kailuo/group_index/data_rel/package.0", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;

    Searcher::Pointer searcher = factory.CreateSearcher();
    searcher->LoadIndex(data, handler->size());
    GroupIvfIndex& index = ((GroupIvfSearcher*)searcher.get())->index_;
    for (size_t i = 0; i < index.group_manager_.group_vec_.size(); i++) {
        //std::cout<< "level:"<<index.group_manager_.group_vec_.at(i).level<<" ;id:"<<index.group_manager_.group_vec_.at(i).id<<std::endl;
    }
    //ASSERT_EQ(3, index.group_manager_.group_vec_.at(0));
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadConstant,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "1000&3:18853#10,13832#10,18876#10,13898#10,18966#10,13932#10,13828#10,18825#10,13803#10,18832#10,13810#10,18875#10,18924#10,18944#10,18822#10,18911#10,18816#10,18961#10,13903#10,18819#10,13876#10,18959#10,6283#10,13928#10,18906#10,18891#10,13908#10,18899#10,18904#10,18934#10,18890#10,18877#10,18859#10,18923#10,13904#10||0.0599496 -0.0426028 0.118882 -0.0821001 -0.172668 0.0365412 -0.0817514 0.0702005 0.0621878 -0.137953 -0.0766542 0.210053 -0.0587268 -0.0188314 -0.0467993 0.011791 -0.0777488 0.117934 -0.0530871 -0.0821527 0.0764689 -0.047727 0.162653 0.0689887 -0.0397429 0.0597757 0.0409169 -0.0275424 0.0371887 -0.00384824 0.1785 0.0155983 -0.054817 -0.158354 -0.00748303 -0.0891769 -0.0234836 0.11721 -0.0139884 -0.123226 0.0520906 0.0852503 -0.157158 0.0773817 0.110088 0.0685822 0.0484215 -0.0831212 0.0932779 0.146491 0.0681295 0.0112493 0.0117855 -0.239697 0.284722 0.12828 0.0685867 0.035939 -0.0114124 0.072579 -0.0822875 0.0176876 0.231602 0.172468 -0.111326 -0.0164234 0.013479 -0.0344876 -0.00484037 -0.00157342 0.0178474 0.0134294 -0.143456 0.00124241 -0.0157692 -0.0957733 -0.0673775 -0.0617966 0.046741 0.0777142 -0.175472 -0.0347203 0.0953245 0.0298437 0.0707871 0.20104 -0.0400624 -0.126165 0.00712993 0.0816294 -0.105352 0.00407734 0.0770842 -0.0159163 -0.0987312 -0.0280873 0.066135 0.111586 0.0411833 0.0975941 -0.0720306 -0.0165915 -0.0768524 0.0661915 -0.0178893 -0.0244516 0.0969599 -0.0295918 0.0318867 0.00612984 -0.238353 0.063868 0.0749163 0.0771469 -0.620371 -0.0714202 0.273015 -0.471776 -0.0488319 -0.00772873 0.0147852 4.40997e-05 -0.360906 -0.0400248 0.24409 0.273994 -0.0766937 -0.162212";
    int ret_code = searcher->Search(search_str, &context);
    std::vector<SearchResult>& results = context.Result();
    //std::sort(results.begin(), results.end(), [](SearchResult& a, SearchResult& b) {
    //return a.score < b.score;
    //});
    size_t count = 0;
    for (auto& r : results) {
        if (r.gloid == 698807) {
            std::cout<<"count "<<count<<" score is "<<r.score<<std::endl;
        }
        count++;
    }

    std::vector<GroupInfo> group_infos;
    std::vector<int> labels;
    index.SearchGroup(698807, group_infos, labels);

    for (auto& info : group_infos) {
        std::cout<<"level "<<info.level<< " id "<<info.id<<std::endl;
    }

    //for (size_t i = 0; i < 10; i++) {
    //  std::cout<<i <<" "<<results.at(i).gloid<< " result score: " << results.at(i).score<<std::endl;
        //if (fabs(0.216464 - results.at(i).score) < 0.000001) {
        //  std::cout<<i <<" "<<results.at(i).gloid<< " result score: " << results.at(i).score<<std::endl;
        //}

    //}
    EXPECT_EQ(1, context.Result().size());

    //auto product_size = index.get_centroid_resource().getIntegrateMeta().fragmentNum * sizeof(uint16_t);
    }*/
MERCURY_NAMESPACE_END(core);
