/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <omp.h>

#define protected public
#define private public
#include "src/core/algorithm/vamana/disk_vamana_index.h"
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class DiskVamanaSingleGraphTest: public testing::Test
{
public:
    void SetUp()
    {
        // test using shoe data:
        base_filename_ = "tests/core/vamana/test_data_shoe_5m/img_shoe_5m.base";
        index_shoe_prefix_ = "tests/core/vamana/test_data_shoe_5m/disk_index_shoe_single";

        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 5000000);
        index_params_.set(PARAM_INDEX_TYPE, "DiskVamana");
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, false);
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 64);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 150);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_ALPHA, 1.2);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION, 3000);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED, false);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/vamana/test_data_shoe_5m/cluster_results");
    }

    void TearDown()
    {

    }

    AlgorithmFactory factory_;
    IndexParams index_params_;

    std::string base_filename_, index_shoe_prefix_;
};

TEST_F(DiskVamanaSingleGraphTest, BuildAndDump) {

    GTEST_SKIP() << "Skipping DiskVamanaSingleGraphTest.BuildAndDump";

    factory_.SetIndexParams(index_params_);

    Index::Pointer index_p = factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    std::ifstream base_file(base_filename_);
    if (!base_file.is_open()) {
        LOG_ERROR("Error: Could not open base_file with path: %s", base_filename_.c_str());
        return;
    }

    size_t num_base = 0;

    std::vector<std::string> base_vecs;
    base_vecs.reserve(index->GetMaxDocNum());
    std::string base_vec;
    while (std::getline(base_file, base_vec, '\x1f')) {
        if (base_vec.length() > 1) {
            base_vecs.push_back(base_vec);
        }
    }

    LOG_INFO("num of base points is %lu", base_vecs.size());

    for (size_t i = 0; i < base_vecs.size(); i++) {
        QueryInfo query_info(base_vecs[i]);
        if (index->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeHalfFloat) {
            query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
        }
        if (!query_info.MakeAsBuilder()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
            return;
        }
        index->BaseIndexAdd(i, 0, query_info.GetVector(), query_info.GetVectorLen());
    }

    index->BuildMemIndex();

    index->DumpMemLocal(index_shoe_prefix_ + ".mem");

    index->CreateDiskLayout(index_shoe_prefix_ + ".mem.data", index_shoe_prefix_ + ".mem", index_shoe_prefix_ + ".disk");

    ASSERT_TRUE(file_exists(index_shoe_prefix_ + ".disk"));

}

TEST_F(DiskVamanaSingleGraphTest, LoadAndSearch) {

    GTEST_SKIP() << "Skipping DiskVamanaSingleGraphTest.LoadAndSearch";

    factory_.SetIndexParams(index_params_);

    Index::Pointer index_p = factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    std::ifstream base_file(base_filename_);
    if (!base_file.is_open()) {
        LOG_ERROR("Error: Could not open base_file with path: %s", base_filename_.c_str());
        return;
    }

    size_t num_base = 0;

    std::vector<std::string> base_vecs;
    base_vecs.reserve(index->GetMaxDocNum());
    std::string base_vec;
    while (std::getline(base_file, base_vec, '\x1f')) {
        if (base_vec.length() > 1) {
            base_vecs.push_back(base_vec);
        }
    }

    LOG_INFO("num of base points is %lu", base_vecs.size());

    for (size_t i = 0; i < base_vecs.size(); i++) {
        QueryInfo query_info(base_vecs[i]);
        if (!query_info.MakeAsBuilder()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
            return;
        }
        index->PQIndexAdd(i, query_info.GetVector());
    }

    ASSERT_TRUE(file_exists(index_shoe_prefix_ + ".disk"));

    index->LoadDiskIndex(index_shoe_prefix_ + ".disk", "");

    std::vector<uint32_t> node_list;

    uint32_t num_nodes_to_cache = 0;

    std::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;

    index->cache_bfs_levels(num_nodes_to_cache, node_list);

    index->load_cache_list(node_list);

    node_list.clear();
    
    node_list.shrink_to_fit();

    std::string search_str = "0.024243 -0.077105 -0.143239 0.128986 -0.052206 0.219534 -0.099307 -0.210907 0.000328 -0.117573 0.04972 -0.0128 -0.061421 0.072023 -0.087146 -0.155934 -0.036539 -0.012248 -0.061297 -0.1019 0.123233 0.068733 -0.070405 0.044882 -0.024621 -0.0153 -0.13201 0.141363 0.039671 -0.162601 -0.072712 -0.001794 0.033505 0.05173 0.141834 0.024119 -0.046238 -0.168437 -0.07072 -0.081426 0.13474 -0.033715 0.006738 0.067471 0.124383 -0.182244 0.051407 0.046778 0.007796 0.016033 -0.018392 0.146075 0.014159 -0.005309 -0.070059 0.027275 -3.3e-05 -0.053808 0.062941 0.129937 0.002475 -0.087751 0.049858 0.019514 -0.13099 0.059999 -0.036173 -0.086178 0.048415 0.052633 -0.061978 -0.02301 -0.133252 -0.090564 0.035663 -0.067442 -0.118479 0.092425 -0.026573 -0.031636 0.165636 0.013061 0.09039 -0.091541 0.017454 0.005997 0.095756 0.084898 -0.225537 -0.01896 0.052134 -0.046275 0.06373 -0.112419 0.014479 -0.091589 0.030476 0.096252 -0.124276 0.064662 0.129112 -0.077564 0.112072 -0.051064 -0.035439 -0.026589 0.162878 -0.019143 0.184887 -0.021283 -0.006504 0.118282 -0.019384 0.145319 -0.06274 -0.079104 -0.091715 -0.038275 -0.020641 -0.075488 -0.00707 -0.007786 0.125436 -0.055758 0.110477 0.02674 -0.022083 -0.104292";

    QueryInfo query_info(search_str);
        
    query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

    if (!query_info.MakeAsSearcher()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
    }

    QueryInfo query_info_raw(query_info.GetRawQuery());
    
    if (!query_info_raw.MakeAsSearcher()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info_raw.GetRawQuery().c_str());
    }

    const uint32_t topk = 300;

    uint32_t L = 350;

    uint32_t optimized_beamwidth = 6;

    auto stats = new QueryStats();

    std::vector<uint64_t> query_result_ids_64(topk);
    std::vector<float> query_result_dist(topk);

    index->cached_beam_search_half(query_info.GetVector(), query_info_raw.GetVector(), topk, L,
                                    query_result_ids_64.data(), query_result_dist.data(),
                                    optimized_beamwidth, stats);

    std::cout << "Search Results:" << std::endl;
    for (size_t i = 0; i < topk; i++) {
        std::cout << "id : " << query_result_ids_64[i] << " ; dist : " << query_result_dist[i] << std::endl;
    }
}

MERCURY_NAMESPACE_END(core);
