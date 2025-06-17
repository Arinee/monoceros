/// Copyright (c) 2020, xiaohongshu Inc. All rights reserved.
/// Author: sunan <sunan@xiaohongshu.com>
/// Created: 2020-08-27 12:00

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

class GroupIvfPqBuilderTest: public testing::Test
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

TEST_F(GroupIvfPqBuilderTest, TestGetRankScore) {
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

TEST_F(GroupIvfPqBuilderTest, TestGetRankScoreBySpecifiedGroupLevel) {
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


/*
TEST_F(GroupIvfPqBuilderTest, TestLocalRank) {
    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/sunan/image_lve/mercury_index.package", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;
    
    mercury::core::GroupIvfPqBuilder builder;
    builder.index_.Load(data, handler->size());

    std::ifstream in("/data1/sunan/cate_vec_0");
    std::ofstream out("/data1/sunan/doc_file_score");
    char buffer[5000];
    if (!in.is_open())
    { std::cout << "Error opening file"; exit (1); }
    while (!in.eof())
    {
        in.getline(buffer, 5000);
        float score = 0.0000;
        std::string buf_str = buffer;
	if (buf_str == "") {
	  continue;
	}
        buf_str = StringUtil::split(buf_str, "=")[1];
        int ret = builder.GetRankScore(buf_str.substr(0, buf_str.size()-1), &score);
        ASSERT_EQ(ret, 0);
        out << buffer <<"=" << score << std::endl;
    }
}
*/

/*
TEST_F(GroupIvfPqBuilderTest, TestLocalLoad) {
    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/sunan/image_lve/mercury_index.package", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());

    std::cout<<"handler size: "<<handler->size()<<std::endl;

    //Searcher::Pointer searcher = factory.CreateBuilder();
    //searcher->LoadIndex(data, handler->size());
    //GroupIvfIndex& index = ((GroupIvfSearcher*)searcher.get())->index_;
    mercury::core::GroupIvfPqBuilder builder;
    builder.index_.Load(data, handler->size());

    LOG_INFO("doc num in this segment is %d", builder.index_.GetPqCodeProfile().getDocNum());
    CoarseIndex<BigBlock>::PostingIterator iter = builder.index_.GetCoarseIndex().search(120);
    while (UNLIKELY(!iter.finish())) {
    	docid_t docid = iter.next();
        std::cout<<docid<<" "<<builder.index_.GetPqCodeProfile().getInfo(docid)<<std::endl;	
    }
}
*/


MERCURY_NAMESPACE_END(core);
