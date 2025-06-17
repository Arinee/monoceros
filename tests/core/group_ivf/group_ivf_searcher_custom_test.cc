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

class GroupIvfSearcherCustomTest: public testing::Test
{
public:
    void SetUp()
        {
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
            index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/custom_distance_data/");
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_METHOD, "IP");
            index_params_.set(PARAM_DIMENSION, 256);
            index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
            index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
            index_params_.set(PARAM_CUSTOMED_PART_DIMENSION, 128);
            factory_.SetIndexParams(index_params_);
        }

    void TearDown()
        {
        }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfSearcherCustomTest, TestSearchCustom) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    // 创建一个GroupIvfIndex，并将参数传入
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
    std::string search_str = "0:0#100||" + data2_str;
    group_str = "1:11;2:200||" + data2_str;
    size_t size2 = core_index2->index_meta_._element_size;
    docid_t docid2 = 0;
    ret = core_index2->Add(docid2, INVALID_PK, group_str);
    data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, 1);
    data2_str = DataToStr(data2);
    group_str = "1:11;2:200||" + data2_str;
    ret = core_index2->Add(docid2 + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index2->GetDocNum());

    MergeIdMap id_map;
    id_map.push_back(std::make_pair(1, 0));
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

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher*)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile,
                            reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(0, context.Result().size());
    // ASSERT_NEAR(-1.28, context.Result().at(0).score, 0.001);
    // ASSERT_NEAR(-0.64, context.Result().at(1).score, 0.001);
    // ASSERT_NEAR(-0.64, context.Result().at(2).score, 0.001);
    // ASSERT_NEAR(-0.64, context.Result().at(3).score, 0.001);
}

MERCURY_NAMESPACE_END(core);
