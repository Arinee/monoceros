// /// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
// /// Author: kailuo <kailuo@xiaohongshu.com>
// /// Created: 2019-12-17 00:59

// #include <gtest/gtest.h>
// #include <iostream>
// #include <fstream>

// #define protected public
// #define private public
// #include "src/core/algorithm/group_ivf/group_ivf_builder.h"
// #include "src/core/algorithm/algorithm_factory.h"
// #include "src/core/algorithm/query_info.h"
// #include "src/core/framework/index_storage.h"
// #include "src/core/framework/instance_factory.h"
// #include "src/core/utils/index_meta_helper.h"
// #undef protected
// #undef private

// MERCURY_NAMESPACE_BEGIN(core);

// class GroupIvfTwoValueVectorTest: public testing::Test
// {
// public:
//     void SetUp()
//     {
//         index_params_.set(PARAM_COARSE_SCAN_RATIO, 0.5);
//         index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/two_value_data");
//         index_params_.set(PARAM_DATA_TYPE, "binary");
//         index_params_.set(PARAM_METHOD, "HAMMING");
//         index_params_.set(PARAM_DIMENSION, 128);
//         index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
//         index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 100);
//         index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
//         factory_.SetIndexParams(index_params_);
//     }

//     void TearDown()
//     {
//     }

//     AlgorithmFactory factory_;
//     IndexParams index_params_;
// };

// TEST_F(GroupIvfTwoValueVectorTest, TestQueryInfo) {

//     std::string data_str1 = "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0";
//     std::string group_str1 = "1:11;2:200||" + data_str1;
//     QueryInfo query_info1(group_str1); // groupinfos 对应的 topks
//     query_info1.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
//     query_info1.MakeAsBuilder();
//     ASSERT_EQ("1:11;2:200", query_info1.GetRawGroupInfo());

//     std::string data_str2 = "1 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0";
//     std::string group_str2 = "1000&3:9654#10||" + data_str2;
//     QueryInfo query_info2(group_str2); // groupinfos 对应的 topks
//     query_info2.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
//     query_info2.MakeAsSearcher();
//     ASSERT_EQ("1000&3:9654#10", query_info2.GetRawGroupInfo());

//     IndexMeta index_meta;
//     if (!IndexMetaHelper::parseFrom("binary", "HAMMING", 32, index_meta)) {
//         LOG_ERROR("Failed to init index meta.");
//     }
//     float distance = index_meta.distance(query_info1.GetVector(), query_info2.GetVector());
//     std::cout << "distance between query1 and query2: " << distance << std::endl;
// }

// TEST_F(GroupIvfTwoValueVectorTest, TestDistanceCalculation) {
//     Builder::Pointer builder_p = factory_.CreateBuilder();
//     mercury::core::GroupIvfBuilder* builder = dynamic_cast<mercury::core::GroupIvfBuilder*>(builder_p.get());

//     ASSERT_TRUE(builder != nullptr);
//     ASSERT_EQ(100, builder->index_->GetMaxDocNum());

//     std::string data_str = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1";
//     std::string group_str = "1:11;2:200||" + data_str;
//     QueryInfo query_info(group_str); // groupinfos 对应的 topks
//     query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
//     query_info.MakeAsBuilder();
    
//     gindex_t group_index = 0;
//     docid_t docid = 0;
//     int ret = builder->AddDoc(docid, INVALID_PK, group_str);
//     ASSERT_EQ(0, ret);

//     const void *dump_content = nullptr;
//     size_t dump_size = 0;

//     dump_content = builder->DumpIndex(&dump_size);
//     ASSERT_TRUE(dump_content != nullptr);
//     ASSERT_TRUE(dump_size != 0);

//     Index::Pointer loaded_index_p = factory_.CreateIndex(true);
//     loaded_index_p->Load(dump_content, dump_size);
//     GroupIvfIndex* core_index = dynamic_cast<GroupIvfIndex*>(loaded_index_p.get());

//     SlotIndex label = 1;
    
//     EXPECT_EQ(label, core_index->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), group_index));
//     EXPECT_EQ(0, core_index->coarse_index_.search(label).next());
//     EXPECT_EQ(1, core_index->GetDocNum());
//     EXPECT_EQ(100, core_index->GetMaxDocNum());
// }

// MERCURY_NAMESPACE_END(core);
