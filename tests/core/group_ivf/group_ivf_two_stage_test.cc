/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-12-17 00:59

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>

#define protected public
#define private public
#include "common.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/group_ivf/group_ivf_builder.h"
#include "src/core/algorithm/group_ivf/group_ivf_merger.h"
#include "src/core/algorithm/group_ivf/group_ivf_searcher.h"
#include "src/core/algorithm/group_ivf/mocked_vector_reader.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfTwoStageTest : public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 1.0);
        index_params_.set(PARAM_FINE_SCAN_RATIO, 1.0);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data_two_stage/");
        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "IP");
        index_params_.set(PARAM_DIMENSION, 64);
        index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        index_params_.set(PARAM_ENABLE_FINE_CLUSTER, true);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
        factory_.SetIndexParams(index_params_);
    }

    void TearDown() {}

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfTwoStageTest, TestBuild)
{
    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::GroupIvfBuilder *builder = dynamic_cast<mercury::core::GroupIvfBuilder *>(builder_p.get());

    ASSERT_TRUE(builder != nullptr);
    ASSERT_EQ(10000, builder->index_->GetMaxDocNum());

    std::string data_str =
        "0.0926264 0.0079578 0.0105586 0.0418111 0.1812230 0.0563133 -0.0899557 0.1065935 -0.0799954 0.0870304 0."
        "0170577 -0.2035020 -0.0675575 -0.0591432 -0.1244585 0.1755335 -0.1775552 -0.0377039 0.0819606 0.0970977 0."
        "0279189 0.0369875 -0.0192580 0.0029287 -0.0357301 0.1562776 0.1798831 -0.1910289 0.1729529 0.1310860 0."
        "0251708 -0.0850961 -0.0402595 0.0584627 0.0042594 0.0411005 -0.0828406 0.0533206 0.1465107 0.0839182 -0."
        "0228154 0.0684170 -0.5612501 0.0055404 -0.0606350 0.0044111 -0.0993920 -0.0074073 0.1077544 0.1560783 0."
        "1391315 0.1358985 -0.0099312 -0.0318797 -0.0148738 -0.0368931 -0.0088994 -0.1230934 0.0407985 0.0562747 0."
        "1737005 0.0836068 0.0971618 0.0989029";
    std::string group_str = "0:0;1:1||" + data_str;

    docid_t docid = 0;
    int ret = builder->AddDoc(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    data_str = "-0.0393940 -0.0504650 -0.0875190 -0.0643030 -0.1378400 -0.0762630 0.0239965 0.0190530 -0.1256575 "
               "-0.2664550 -0.0005035 -0.2034175 0.1323655 0.0094145 0.0743665 0.0673560 0.0073090 -0.0673595 "
               "-0.0890205 -0.1717630 0.0272180 0.0023510 -0.0747015 -0.0523320 -0.1239750 -0.2387470 0.0524925 "
               "-0.0582860 0.0732185 -0.0070135 0.1633875 -0.0867170 0.0198160 -0.0796320 -0.0304365 0.1818590 "
               "-0.2894615 -0.0979785 0.0475245 0.1159425 0.0018870 0.1361200 -0.5312295 0.0517105 0.0648055 0.1056075 "
               "0.0348860 -0.1045560 -0.0454515 -0.0255400 0.0623950 -0.0347220 -0.1781800 -0.1179305 0.0186210 "
               "0.0569400 0.0451100 0.0448940 0.0195640 0.0547495 0.1084105 0.0080165 -0.0956595 -0.0593615";
    group_str = "0:0;1:1||" + data_str;
    ret = builder->AddDoc(docid, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);

    const void *dump_content = nullptr;
    size_t dump_size = 0;

    dump_content = builder->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_content != nullptr);
    ASSERT_TRUE(dump_size != 0);

    Index::Pointer loaded_index_p = factory_.CreateIndex(true);
    loaded_index_p->Load(dump_content, dump_size);
    GroupIvfIndex *core_index = dynamic_cast<GroupIvfIndex *>(loaded_index_p.get());

    QueryInfo query_info(group_str);
    query_info.SetFeatureTypes(mercury::core::IndexMeta::kTypeHalfFloat);
    ASSERT_TRUE(query_info.MakeAsBuilder());
    EXPECT_EQ(3, core_index->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), 0,
                                                  core_index->GetCentroidResourceManager()));
    EXPECT_EQ(303, core_index->GetNearestGroupLabel(query_info.GetVector(), query_info.GetVectorLen(), 3,
                                                  core_index->GetFineCentroidResourceManager()));
    EXPECT_EQ(2, core_index->GetDocNum());
    EXPECT_EQ(10000, core_index->GetMaxDocNum());
}

TEST_F(GroupIvfTwoStageTest, TestSearch)
{
    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index1 = dynamic_cast<mercury::core::GroupIvfIndex *>(index1.get());

    std::string data_str =
        "0.0926264 0.0079578 0.0105586 0.0418111 0.1812230 0.0563133 -0.0899557 0.1065935 -0.0799954 0.0870304 0."
        "0170577 -0.2035020 -0.0675575 -0.0591432 -0.1244585 0.1755335 -0.1775552 -0.0377039 0.0819606 0.0970977 0."
        "0279189 0.0369875 -0.0192580 0.0029287 -0.0357301 0.1562776 0.1798831 -0.1910289 0.1729529 0.1310860 0."
        "0251708 -0.0850961 -0.0402595 0.0584627 0.0042594 0.0411005 -0.0828406 0.0533206 0.1465107 0.0839182 -0."
        "0228154 0.0684170 -0.5612501 0.0055404 -0.0606350 0.0044111 -0.0993920 -0.0074073 0.1077544 0.1560783 0."
        "1391315 0.1358985 -0.0099312 -0.0318797 -0.0148738 -0.0368931 -0.0088994 -0.1230934 0.0407985 0.0562747 0."
        "1737005 0.0836068 0.0971618 0.0989029";
    std::string group_str = "0:0;1:1||" + data_str;
    size_t size = core_index1->index_meta_._element_size;
    docid_t docid = 0;
    int ret = core_index1->Add(docid, INVALID_PK, group_str);
    ret = core_index1->Add(docid + 1, INVALID_PK, group_str);
    ASSERT_EQ(0, ret);
    ASSERT_EQ(2, core_index1->GetDocNum());

    Index::Pointer index2 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index2 = dynamic_cast<mercury::core::GroupIvfIndex *>(index2.get());

    std::string data2_str = "-0.0393940 -0.0504650 -0.0875190 -0.0643030 -0.1378400 -0.0762630 0.0239965 0.0190530 -0.1256575 "
               "-0.2664550 -0.0005035 -0.2034175 0.1323655 0.0094145 0.0743665 0.0673560 0.0073090 -0.0673595 "
               "-0.0890205 -0.1717630 0.0272180 0.0023510 -0.0747015 -0.0523320 -0.1239750 -0.2387470 0.0524925 "
               "-0.0582860 0.0732185 -0.0070135 0.1633875 -0.0867170 0.0198160 -0.0796320 -0.0304365 0.1818590 "
               "-0.2894615 -0.0979785 0.0475245 0.1159425 0.0018870 0.1361200 -0.5312295 0.0517105 0.0648055 0.1056075 "
               "0.0348860 -0.1045560 -0.0454515 -0.0255400 0.0623950 -0.0347220 -0.1781800 -0.1179305 0.0186210 "
               "0.0569400 0.0451100 0.0448940 0.0195640 0.0547495 0.1084105 0.0080165 -0.0956595 -0.0593615";
    group_str = "0:0;1:1||" + data2_str;
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

    GroupIvfMerger *ivf_merger = dynamic_cast<GroupIvfMerger *>(index_merger.get());
    mercury::core::GroupIvfIndex &merged_index = ivf_merger->merged_index_;
    ASSERT_EQ(3, merged_index.GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void *dump_data = index_merger->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    int ret_code = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret_code);

    ASSERT_EQ(index_searcher->getFType(), IndexMeta::kTypeHalfFloat);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    mercury::core::AttrRetriever retriever;
    mercury::core::MockedVectorReader reader(((GroupIvfSearcher *)index_searcher.get())->index_);
    retriever.set(std::bind(&MockedVectorReader::ReadProfile, reader, std::placeholders::_1, std::placeholders::_2));
    context.setAttrRetriever(retriever);
    std::string search_str = "0:0#100||" + data_str;
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(mercury::core::IndexMeta::kTypeHalfFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret_code = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(3, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_EQ(2, context.Result().at(2).gloid);
}

MERCURY_NAMESPACE_END(core);
