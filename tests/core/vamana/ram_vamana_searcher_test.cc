/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>

#define protected public
#define private public
#include "src/core/algorithm/vamana/ram_vamana_searcher.h"
#include "src/core/algorithm/vamana/ram_vamana_builder.h"
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class RamVamanaSearcherTest: public testing::Test
{
public:
    void SetUp()
    {
        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";
        ram_vamana_index_sift_learn_prefix_ = "tests/core/vamana/test_data/ram_vamana_half_index_sift_learn_R32_L50_A1.2";

        size_t data_num, data_dim;
        get_bin_metadata(base_filename_, data_num, data_dim);

        sift_index_params_.set(PARAM_DATA_TYPE, "half");
        sift_index_params_.set(PARAM_METHOD, "L2");
        sift_index_params_.set(PARAM_DIMENSION, data_dim);

        sift_index_params_.set(PARAM_INDEX_TYPE, "RamVamana");
        sift_index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        sift_index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, data_num);

    }

    void TearDown()
    {

    }

    AlgorithmFactory sift_factory_;
    IndexParams sift_index_params_;

    std::string base_filename_, query_filename_, gt_filename_, ram_vamana_index_sift_learn_prefix_;
};

TEST_F(RamVamanaSearcherTest, TestLoadAndSearch) {

    sift_factory_.SetIndexParams(sift_index_params_);

    Index::Pointer index_p = sift_factory_.CreateIndex();

    mercury::core::RamVamanaIndex* index = dynamic_cast<mercury::core::RamVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    auto index_mata = index->GetIndexMeta();

    size_t ori_data_size = 4;

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
        reader.read((char *)data.get(), dim * ori_data_size);
        std::string build_str;
        for (size_t j = 0; j < dim; j++)
        {
            build_str += std::to_string(data[j]);
            if (j != dim - 1) {
                build_str += " ";
            }
        }
        QueryInfo query_info(build_str);

        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

        if (!query_info.MakeAsBuilder()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
            return;
        }

        index->BaseIndexAdd(i, query_info.GetVector());
    }

    index->BuildMemIndex();

    index->DumpMemLocal(ram_vamana_index_sift_learn_prefix_);

    ASSERT_TRUE(file_exists(ram_vamana_index_sift_learn_prefix_));

    ASSERT_TRUE(file_exists(ram_vamana_index_sift_learn_prefix_ + ".data"));

    ASSERT_EQ(get_file_size(ram_vamana_index_sift_learn_prefix_), 13200024);

    const void *dump_data = nullptr;

    size_t dump_size = 0;

    index->Dump(dump_data, dump_size);

    Searcher::Pointer searcher_p = sift_factory_.CreateSearcher();

    searcher_p->LoadIndex(dump_data, dump_size);

    searcher_p->LoadIndex(ram_vamana_index_sift_learn_prefix_);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    std::string search_str = "100&0:0#100||1.00 3.00 11.00 110.00 62.00 22.00 4.00 0.00 43.00 21.00 22.00 18.00 6.00 28.00 64.00 9.00 11.00 1.00 0.00 0.00 1.00 40.00 101.00 21.00 20.00 2.00 4.00 2.00 2.00 9.00 18.00 35.00 1.00 1.00 7.00 25.00 108.00 116.00 63.00 2.00 0.00 0.00 11.00 74.00 40.00 101.00 116.00 3.00 33.00 1.00 1.00 11.00 14.00 18.00 116.00 116.00 68.00 12.00 5.00 4.00 2.00 2.00 9.00 102.00 17.00 3.00 10.00 18.00 8.00 15.00 67.00 63.00 15.00 0.00 14.00 116.00 80.00 0.00 2.00 22.00 96.00 37.00 28.00 88.00 43.00 1.00 4.00 18.00 116.00 51.00 5.00 11.00 32.00 14.00 8.00 23.00 44.00 17.00 12.00 9.00 0.00 0.00 19.00 37.00 85.00 18.00 16.00 104.00 22.00 6.00 2.00 26.00 12.00 58.00 67.00 82.00 25.00 12.00 2.00 2.00 25.00 18.00 8.00 2.00 19.00 42.00 48.00 11.00||mercury.vamana.index.search.l=200";
    QueryInfo query_info(search_str);
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    int ret_code = searcher_p->Search(query_info, &context);
    ASSERT_EQ(ret_code, 0);
    std::cout << "context.Result().size(): " << context.Result().size() << std::endl;
    for (SearchResult result : context.Result()) {
        std::cout << "id = " << result.gloid << " ; " << "score = " << result.score << std::endl;
    }
    context.clean();

    index_p.reset();
}

MERCURY_NAMESPACE_END(core);
