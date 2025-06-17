/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-02 17:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_builder.h"
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_searcher.h"
#include "src/core/algorithm/group_ivf_pq/group_ivf_pq_merger.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "../group_ivf/common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/utils/string_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfPqIPBuilderTest: public testing::Test
{
public:
    void SetUp()
        {
            std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
            index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
            index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf_pq/test_data/");
            index_params_.set(PARAM_DATA_TYPE, "float");
            index_params_.set(PARAM_METHOD, "IP");
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

TEST_F(GroupIvfPqIPBuilderTest, TestGetRankScore) {
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index.get());
    
    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::GroupIvfPqBuilder* builder = dynamic_cast<mercury::core::GroupIvfPqBuilder*>(builder_p.get());

    const void *data1 = core_index->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);
    std::string data_str = DataToStr(data1);
    float score = 0.0;
    int ret = builder->GetRankScore(data_str, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 2);

    std::string cate_data_str = "1:11||" + DataToStr(data1);
    builder->GetRankScore(cate_data_str, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 0);

    const void *data2 = core_index->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    std::string data_str2 = DataToStr(data2);
    ret = builder->GetRankScore(data_str2, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 1);
}

TEST_F(GroupIvfPqIPBuilderTest, TestGetRankScoreBySpecifiedGroupLevel) {
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 1);
    index_params.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf_pq/test_data/");
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, "GroupIvfPq");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
    index_params.set(PARAM_PQ_SCAN_NUM, 5000);
    index_params.set(PARAM_SORT_BUILD_GROUP_LEVEL, 1);
    mercury::core::GroupIvfPqBuilder builder;
    builder.Init(index_params);
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 0));
    ASSERT_FLOAT_EQ(0, builder.CalcScore(0, 8191));
    ASSERT_FLOAT_EQ(1, builder.CalcScore(1, 0));
    ASSERT_FLOAT_EQ(8192, builder.CalcScore(1, 8191));
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::GroupIvfPqIndex* core_index = dynamic_cast<mercury::core::GroupIvfPqIndex*>(index.get());
    const void *data1 = core_index->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 1);
    std::string data_str = DataToStr(data1);
    float score = 0.0;
    int ret = builder.GetRankScore(data_str, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 0);

    std::string cate_data_str = "1:11||" + DataToStr(data1);
    builder.GetRankScore(cate_data_str, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 7);

    const void *data2 = core_index->centroid_resource_manager_.GetCentroidResource(0).getValueInRoughMatrix(0, 0);
    std::string data_str2 = DataToStr(data2);
    ret = builder.GetRankScore(data_str2, &score);
    ASSERT_EQ(ret, 0);
    ASSERT_FLOAT_EQ(score, 0);
}

MERCURY_NAMESPACE_END(core);
