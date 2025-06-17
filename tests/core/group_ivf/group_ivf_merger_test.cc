/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/algorithm/group_ivf/group_ivf_merger.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "common.h"
#include "src/butil/time.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfMergerTest: public testing::Test
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

    void TearDown()
    {
    }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfMergerTest, TestMerge) {
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
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1, core_index1->GetDocNum());

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfIndex*>(index2.get());

    SlotIndex label2 = 1;
    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "0:0;1:11;2:200||" + data2_str;
    size_t size2 = core_index2->index_meta_._element_size;
    ret = core_index2->Add(0, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1, core_index2->GetDocNum());
    group_str = "1:1111||" + data2_str;
    ret = core_index2->Add(1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index2->GetDocNum());

    MergeIdMap id_map;
    id_map.push_back(std::make_pair(1, 0));
    id_map.push_back(std::make_pair(1, 1));
    id_map.push_back(std::make_pair(0, 0));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();

    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);

    GroupIvfMerger* ivf_merger = dynamic_cast<GroupIvfMerger*>(index_merger.get());
    mercury::core::GroupIvfIndex& merged_index = ivf_merger->merged_index_;

    ASSERT_EQ(3, merged_index.GetMaxDocNum());
    ASSERT_EQ(3, merged_index.GetDocNum());

    //测试倒排
    ASSERT_EQ(10, merged_index.coarse_index_.getHeader()->slotNum);
    std::vector<uint32_t> coarseIndexLabels;
    coarseIndexLabels.push_back(0);
    coarseIndexLabels.push_back(1);
    for (const auto& e : coarseIndexLabels) {
        mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(e);
        while (!iter.finish()) {
            std::cout<<"label:"<<e<<"; docid:"<<iter.next()<<std::endl;
        }
    }

    mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(0);
    ASSERT_TRUE(iter.finish());
    // ASSERT_EQ(0, iter.next());

    iter = merged_index.coarse_index_.search(1);
    ASSERT_EQ(0, iter.next());
    ASSERT_EQ(2, iter.next());
}

TEST_F(GroupIvfMergerTest, TestMultiAgeMerge) {
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 0.5);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    index_params.set(PARAM_MULTI_AGE_MODE, true);
    AlgorithmFactory factory;
    factory.SetIndexParams(index_params);

    Index::Pointer index1 = factory.CreateIndex();
    mercury::core::GroupIvfIndex* core_index1 = dynamic_cast<mercury::core::GroupIvfIndex*>(index1.get());

    gindex_t group_index = 0;
    SlotIndex label = 1;
    const void *data = core_index1->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label);
    std::string data_str = DataToStr(data);
    std::string group_str = "0:0;1:11;2:200||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int64_t now_timestamp_s = butil::gettimeofday_s();
    char hexStr[9];
    snprintf(hexStr, 9, "%08lx", now_timestamp_s);
    std::string mock_note_id = hexStr;
    mock_note_id += "abcdefghigklmnop";
    int ret = core_index1->Add(docid, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1, core_index1->GetDocNum());

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex* core_index2 = dynamic_cast<mercury::core::GroupIvfIndex*>(index2.get());

    SlotIndex label2 = 1;
    const void *data2 = core_index2->centroid_resource_manager_.GetCentroidResource(group_index)
        .getValueInRoughMatrix(0, label2);

    std::string data2_str = DataToStr(data2);
    group_str = "0:0;1:11;2:200||" + data2_str;
    size_t size2 = core_index2->index_meta_._element_size;
    ret = core_index2->Add(0, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(1, core_index2->GetDocNum());
    group_str = "1:1111||" + data2_str;
    ret = core_index2->Add(1, INVALID_PK, group_str, mock_note_id);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index2->GetDocNum());

    MergeIdMap id_map;
    id_map.push_back(std::make_pair(1, 0));
    id_map.push_back(std::make_pair(1, 1));
    id_map.push_back(std::make_pair(0, 0));

    std::vector<Index::Pointer> indexes;
    indexes.push_back(index1);
    indexes.push_back(index2);

    Merger::Pointer index_merger = factory_.CreateMerger();

    int suc = index_merger->MergeByDocid(indexes, id_map);
    ASSERT_EQ(0, suc);

    GroupIvfMerger* ivf_merger = dynamic_cast<GroupIvfMerger*>(index_merger.get());
    mercury::core::GroupIvfIndex& merged_index = ivf_merger->merged_index_;

    ASSERT_EQ(3, merged_index.GetMaxDocNum());
    ASSERT_EQ(3, merged_index.GetDocNum());

    //测试倒排
    ASSERT_EQ(10, merged_index.coarse_index_.getHeader()->slotNum);
    std::vector<uint32_t> coarseIndexLabels;
    coarseIndexLabels.push_back(0);
    coarseIndexLabels.push_back(1);
    for (const auto& e : coarseIndexLabels) {
        mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(e);
        while (!iter.finish()) {
            std::cout<<"label:"<<e<<"; docid:"<<iter.next()<<std::endl;
        }
    }

    mercury::core::CoarseIndex<SmallBlock>::PostingIterator iter = merged_index.coarse_index_.search(0);
    ASSERT_TRUE(iter.finish());
    // ASSERT_EQ(0, iter.next());

    iter = merged_index.coarse_index_.search(1);
    ASSERT_EQ(0, iter.next());
    ASSERT_EQ(2, iter.next());
}

MERCURY_NAMESPACE_END(core);
