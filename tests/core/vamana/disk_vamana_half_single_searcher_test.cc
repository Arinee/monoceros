/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>

#define protected public
#define private public
#include "src/core/algorithm/vamana/disk_vamana_searcher.h"
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class DiskVamanaHalfSingleSearcherTest: public testing::Test
{
public:
    void SetUp()
    {
        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";

        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);

        index_params_.set(PARAM_INDEX_TYPE, "DiskVamana");
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 1000000);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/vamana/test_data/cluster_results");

        factory_.SetIndexParams(index_params_);

        _data_size = 4;
    }

    void TearDown()
    {

    }

    AlgorithmFactory factory_;
    IndexParams index_params_;

    size_t _data_size;

    std::string base_filename_, query_filename_, gt_filename_;
};

TEST_F(DiskVamanaHalfSingleSearcherTest, TestLoadAndSearch) {

    Index::Pointer index_p = factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(base_filename_.c_str(), std::ios::binary);
    reader.seekg(0, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (unsigned)npts_i32;
    size_t dim = (unsigned)dim_i32;

    std::unique_ptr<float[]> data = std::make_unique<float[]>(dim);

    for (size_t i = 0; i < npts; i++)
    {
        reader.read((char *)data.get(), dim * _data_size);
        std::string build_str;
        for (size_t j = 0; j < dim; j++)
        {
            build_str += std::to_string(data[j]);
            if (j != dim - 1) {
                build_str += " ";
            }
        }
        QueryInfo query_info(build_str);

        if (!query_info.MakeAsBuilder()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
            return;
        }

        index->PQIndexAdd(i, query_info.GetVector());
    }

    const void *dump_data = nullptr;

    size_t dump_size = 0;

    index->Dump(dump_data, dump_size);

    Searcher::Pointer searcher_p = factory_.CreateSearcher();

    searcher_p->LoadIndex(dump_data, dump_size);

    std::string disk_index_sift_learn_path = std::string(get_current_dir_name()) + "/vamana_half_disk.index";

    searcher_p->LoadDiskIndex(disk_index_sift_learn_path, "");

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    std::string search_str = "100&0:0#100||1.00 3.00 11.00 110.00 62.00 22.00 4.00 0.00 43.00 21.00 22.00 18.00 6.00 28.00 64.00 9.00 11.00 1.00 0.00 0.00 1.00 40.00 101.00 21.00 20.00 2.00 4.00 2.00 2.00 9.00 18.00 35.00 1.00 1.00 7.00 25.00 108.00 116.00 63.00 2.00 0.00 0.00 11.00 74.00 40.00 101.00 116.00 3.00 33.00 1.00 1.00 11.00 14.00 18.00 116.00 116.00 68.00 12.00 5.00 4.00 2.00 2.00 9.00 102.00 17.00 3.00 10.00 18.00 8.00 15.00 67.00 63.00 15.00 0.00 14.00 116.00 80.00 0.00 2.00 22.00 96.00 37.00 28.00 88.00 43.00 1.00 4.00 18.00 116.00 51.00 5.00 11.00 32.00 14.00 8.00 23.00 44.00 17.00 12.00 9.00 0.00 0.00 19.00 37.00 85.00 18.00 16.00 104.00 22.00 6.00 2.00 26.00 12.00 58.00 67.00 82.00 25.00 12.00 2.00 2.00 25.00 18.00 8.00 2.00 19.00 42.00 48.00 11.00";
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    int ret_code = searcher_p->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    ASSERT_EQ(context.Result().size(), 100);
    context.clean();

    IndexParams index_params1;
    mercury::core::GeneralSearchContext context1(index_params1);
    std::string search_str1 = "140&0:0#140||1.00 3.00 11.00 110.00 62.00 22.00 4.00 0.00 43.00 21.00 22.00 18.00 6.00 28.00 64.00 9.00 11.00 1.00 0.00 0.00 1.00 40.00 101.00 21.00 20.00 2.00 4.00 2.00 2.00 9.00 18.00 35.00 1.00 1.00 7.00 25.00 108.00 116.00 63.00 2.00 0.00 0.00 11.00 74.00 40.00 101.00 116.00 3.00 33.00 1.00 1.00 11.00 14.00 18.00 116.00 116.00 68.00 12.00 5.00 4.00 2.00 2.00 9.00 102.00 17.00 3.00 10.00 18.00 8.00 15.00 67.00 63.00 15.00 0.00 14.00 116.00 80.00 0.00 2.00 22.00 96.00 37.00 28.00 88.00 43.00 1.00 4.00 18.00 116.00 51.00 5.00 11.00 32.00 14.00 8.00 23.00 44.00 17.00 12.00 9.00 0.00 0.00 19.00 37.00 85.00 18.00 16.00 104.00 22.00 6.00 2.00 26.00 12.00 58.00 67.00 82.00 25.00 12.00 2.00 2.00 25.00 18.00 8.00 2.00 19.00 42.00 48.00 11.00";
    QueryInfo query_info1(search_str1);
    query_info1.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    ASSERT_TRUE(query_info1.MakeAsSearcher());
    int ret_code1 = searcher_p->Search(query_info1, &context1);
    ASSERT_EQ(ret_code1, 0);
    ASSERT_EQ(context1.Result().size(), 140);
    std::cout << "context.Result().size(): " << context1.Result().size() << std::endl;
    for (SearchResult result : context1.Result()) {
        std::cout << "id = " << result.gloid << " ; " << "score = " << result.score << std::endl;
    }
    context1.clean();

    IndexParams index_params2;
    mercury::core::GeneralSearchContext context2(index_params2);
    std::string search_str2 = "200&0:0#200||1.00 3.00 11.00 110.00 62.00 22.00 4.00 0.00 43.00 21.00 22.00 18.00 6.00 28.00 64.00 9.00 11.00 1.00 0.00 0.00 1.00 40.00 101.00 21.00 20.00 2.00 4.00 2.00 2.00 9.00 18.00 35.00 1.00 1.00 7.00 25.00 108.00 116.00 63.00 2.00 0.00 0.00 11.00 74.00 40.00 101.00 116.00 3.00 33.00 1.00 1.00 11.00 14.00 18.00 116.00 116.00 68.00 12.00 5.00 4.00 2.00 2.00 9.00 102.00 17.00 3.00 10.00 18.00 8.00 15.00 67.00 63.00 15.00 0.00 14.00 116.00 80.00 0.00 2.00 22.00 96.00 37.00 28.00 88.00 43.00 1.00 4.00 18.00 116.00 51.00 5.00 11.00 32.00 14.00 8.00 23.00 44.00 17.00 12.00 9.00 0.00 0.00 19.00 37.00 85.00 18.00 16.00 104.00 22.00 6.00 2.00 26.00 12.00 58.00 67.00 82.00 25.00 12.00 2.00 2.00 25.00 18.00 8.00 2.00 19.00 42.00 48.00 11.00";
    QueryInfo query_info2(search_str2);
    query_info2.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    ASSERT_TRUE(query_info2.MakeAsSearcher());
    int ret_code2 = searcher_p->Search(query_info2, &context2);
    ASSERT_EQ(ret_code2, 0);
    ASSERT_EQ(context2.Result().size(), 140);
    context2.clean();

    index_p.reset();

}

MERCURY_NAMESPACE_END(core);
