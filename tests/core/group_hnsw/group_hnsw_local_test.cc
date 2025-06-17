/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-12-17 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/algorithm/group_hnsw/group_hnsw_builder.h"
#include "src/core/algorithm/group_hnsw/group_hnsw_searcher.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupHnswBuilderTest: public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "IP");
        index_params_.set(PARAM_DIMENSION, 128);
        index_params_.set(PARAM_GROUP_HNSW_BUILD_THRESHOLD, 10000);
        index_params_.set(PARAM_HNSW_BUILDER_MAX_LEVEL, 7);
        index_params_.set(PARAM_HNSW_BUILDER_SCALING_FACTOR, 10);
        index_params_.set(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT, 60);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 3200000);
        index_params_.set(PARAM_GRAPH_COMMON_SEARCH_STEP, 30);
        index_params_.set(PARAM_HNSW_BUILDER_EFCONSTRUCTION, 600);
        index_params_.set(PARAM_GRAPH_COMMON_MAX_SCAN_NUM, 10000);
        index_params_.set(PARAM_INDEX_TYPE, "GroupHnsw");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);

        factory_.SetIndexParams(index_params_);
    }

    void TearDown()
    {
    }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupHnswBuilderTest, TestInit) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    mercury::core::GroupHnswSearcher* searcher = dynamic_cast<mercury::core::GroupHnswSearcher *> (index_searcher.get());
    ASSERT_TRUE(searcher != nullptr);

    IndexStorage::Pointer index_storage = InstanceFactory::CreateStorage("MMapFileStorage");
    ASSERT_TRUE(index_storage);
    IndexStorage::Handler::Pointer handler = index_storage->open("/data1/heiniao/data/lve-base-v2-mercury-index.package", false);
    ASSERT_TRUE(handler);
    void *data = nullptr;
    ASSERT_EQ(handler->read((const void **)(&data), handler->size()), handler->size());
    int ret_code = searcher->LoadIndex(data, handler->size());

    // ret_code = searcher->LoadIndex(dump_content, dump_size);
    ASSERT_EQ(ret_code, 0);

    for (size_t i = 0; i < 100; i++) {
        searcher->TestIsContinuousInBruteLevel3Group(i);
    }
    

    // IndexParams index_params;
    // mercury::core::GeneralSearchContext context(index_params);

    // std::string search_str = "6&1:11#3;2:200#3||0.037105188 -0.09963768 0.112405114 -0.09919526 -0.16488822 0.05997022 -0.07301077 -0.03831501 -0.09299047 0.08260672 -0.09259026 0.006538624 -0.12414588 -0.10387742 -0.07497343 -0.09863809 0.024261214 0.0544381 -0.056869343 0.05104494 0.046774007 -0.0647179 0.00081528147 -0.04286386 -0.016324522 0.05661807 -0.2033659 -0.044357743 -0.09281927 0.0997845 -0.14933874 -0.008094892 -0.11624634 -0.14089273 -0.10945671 0.055918273 0.09396473 -0.08005455 0.084400766 0.021296086 0.0609102 0.046105556 0.029863087 -0.099661954 -0.053571478 -0.017541459 0.12246048 -0.14319223 0.0168707 0.14629889 -0.04741165 0.05754831 -0.029780127 0.024464356 0.101223454 -0.09036789 -0.015348834 -0.11011469 -0.12878184 0.11340711 0.09643751 0.08178696 0.10965163 0.06346747 0.043291084 0.108630866 0.116031766 -0.0697338 0.031178674 0.073483236 -0.04411992 0.10767264 0.047490753 -0.09134567 -0.07483047 -0.17655163 0.13136609 0.09006917 0.13187997 -0.12964027 0.09153519 -0.01721974 0.0046212524 0.14212918 0.055485837 -0.19900955 -0.057212397 -0.07277228 -0.016863713 0.053140633 -0.034304697 -0.1115871 -0.15025508 0.18265297 0.110716335 0.03237194 0.007220623 -0.08377357 0.11972804 -0.07653724 0.12246349 -0.086648464 0.12819062 -0.08388696 -0.089263484 0.053282235 0.17635006 -0.0020986022 0.110868335 -0.016155364 -0.11890439 0.10561465 -0.11130717 0.083681956 -0.088963106 0.068944745 0.22988935 0.29236725 -0.078875355 0.46416578 5.1698185e-06 9.277926e-06 4.022857e-05 1.0294868e-05 -4.4157837e-06 -0.0008374268 0.00049100793 0.0004504403";
    // ret_code = searcher->Search(search_str, &context);
    // ASSERT_EQ(ret_code, 0);
    // std::cout << "context.Result().size(): " << context.Result().size() << std::endl;

    // context.clean();
    // search_str = "6&0:0#3||0.037105188 -0.09963768 0.112405114 -0.09919526 -0.16488822 0.05997022 -0.07301077 -0.03831501 -0.09299047 0.08260672 -0.09259026 0.006538624 -0.12414588 -0.10387742 -0.07497343 -0.09863809 0.024261214 0.0544381 -0.056869343 0.05104494 0.046774007 -0.0647179 0.00081528147 -0.04286386 -0.016324522 0.05661807 -0.2033659 -0.044357743 -0.09281927 0.0997845 -0.14933874 -0.008094892 -0.11624634 -0.14089273 -0.10945671 0.055918273 0.09396473 -0.08005455 0.084400766 0.021296086 0.0609102 0.046105556 0.029863087 -0.099661954 -0.053571478 -0.017541459 0.12246048 -0.14319223 0.0168707 0.14629889 -0.04741165 0.05754831 -0.029780127 0.024464356 0.101223454 -0.09036789 -0.015348834 -0.11011469 -0.12878184 0.11340711 0.09643751 0.08178696 0.10965163 0.06346747 0.043291084 0.108630866 0.116031766 -0.0697338 0.031178674 0.073483236 -0.04411992 0.10767264 0.047490753 -0.09134567 -0.07483047 -0.17655163 0.13136609 0.09006917 0.13187997 -0.12964027 0.09153519 -0.01721974 0.0046212524 0.14212918 0.055485837 -0.19900955 -0.057212397 -0.07277228 -0.016863713 0.053140633 -0.034304697 -0.1115871 -0.15025508 0.18265297 0.110716335 0.03237194 0.007220623 -0.08377357 0.11972804 -0.07653724 0.12246349 -0.086648464 0.12819062 -0.08388696 -0.089263484 0.053282235 0.17635006 -0.0020986022 0.110868335 -0.016155364 -0.11890439 0.10561465 -0.11130717 0.083681956 -0.088963106 0.068944745 0.22988935 0.29236725 -0.078875355 0.46416578 5.1698185e-06 9.277926e-06 4.022857e-05 1.0294868e-05 -4.4157837e-06 -0.0008374268 0.00049100793 0.0004504403";
    // ret_code = searcher->Search(search_str, &context);
    // ASSERT_EQ(ret_code, 0);
    // std::cout << "context.Result().size(): " << context.Result().size() << std::endl;
    // // std::free(const_cast<void *>(dump_content));

    // // std::string query_0_0 = "/data0/heiniao/data/hnsw/query_0_0";
    // // std::ifstream query_0_0_file(query_0_0.c_str());
    // // std::string search_str;
    // // while(getline(query_0_0_file, search_str)) {
    // //     // line.pop_back();
    // //     context.clean();
    // //     searcher->Search(search_str, &context);
    // // }


    ASSERT_TRUE(false);
}

MERCURY_NAMESPACE_END(core);
