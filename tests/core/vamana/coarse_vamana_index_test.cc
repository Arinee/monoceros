/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>

#define protected public
#define private public
#include "src/core/algorithm/vamana/coarse_vamana_index.h"
#include "src/core/algorithm/query_info.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class CoarseVamanaIndexTest: public testing::Test
{
public:
    void SetUp()
    {
        R = 32;
        L = 50;
        num_threads = 8;
        alpha = 1.2;
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";
        index_path_prefix_ = "tests/core/vamana/test_data/coarse_in_mem_index_sift_learn_R32_L50_A1.2";
        data_size_ = 4;
        _method = IndexDistance::kMethodFloatSquaredEuclidean;
        half_index_path_prefix_ = "tests/core/vamana/test_data/coarse_in_mem_half_index_sift_learn_R32_L50_A1.2";
        half_data_size_ = 2;
        _half_method = IndexDistance::kMethodHalfFloatSquaredEuclidean;
    }

    void TearDown()
    {
    }

    std::string base_filename_, index_path_prefix_, half_index_path_prefix_, query_filename_, gt_filename_;
    uint16_t data_size_;
    uint16_t half_data_size_;
    uint32_t R, L, num_threads;
    float alpha;
    IndexDistance::Methods _method;
    IndexDistance::Methods _half_method;
};

TEST_F(CoarseVamanaIndexTest, TestBuild) {
    size_t data_num, data_dim;
    get_bin_metadata(base_filename_, data_num, data_dim);
    auto index_build_params = IndexWriteParametersBuilder(L, R)
                                .with_alpha(alpha)
                                .with_saturate_graph(true)
                                .with_num_threads(num_threads)
                                .build();

    auto config = IndexConfigBuilder()
                    .with_dimension(data_dim)
                    .with_max_points(data_num)
                    .is_dynamic_index(false)
                    .with_index_write_params(index_build_params)
                    .is_enable_tags(false)
                    .build();

    size_t num_points = config.max_points + config.num_frozen_pts;
    size_t dim = config.dimension;
    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
            (config.index_write_params == nullptr ? 0 : config.index_write_params->max_degree));
    std::unique_ptr<InMemDataStore> in_mem_data_store_ = std::make_unique<InMemDataStore>(num_points, dim, data_size_);
    in_mem_data_store_->setMethod(_method);
    std::unique_ptr<InMemGraphStore> in_mem_graph_store_ = std::make_unique<InMemGraphStore>(num_points, max_reserve_degree);
    std::unique_ptr<CoarseVamanaIndex> vamana_index = std::make_unique<CoarseVamanaIndex>(data_size_, config, 
                                                        std::move(in_mem_data_store_), std::move(in_mem_graph_store_), false);
    vamana_index->build(base_filename_.c_str(), data_num);
    vamana_index->save(index_path_prefix_.c_str());
    vamana_index.reset();
    ASSERT_TRUE(file_exists(index_path_prefix_));
    ASSERT_EQ(get_file_size(index_path_prefix_), 13200024);
    ASSERT_TRUE(file_exists(std::string(index_path_prefix_) + ".data"));
    ASSERT_EQ(get_file_size(std::string(index_path_prefix_) + ".data"), 51200008);
}

TEST_F(CoarseVamanaIndexTest, TestSearch) {
    ASSERT_TRUE(file_exists(index_path_prefix_));
    ASSERT_EQ(get_file_size(index_path_prefix_), 13200024);
    ASSERT_TRUE(file_exists(std::string(index_path_prefix_) + ".data"));
    ASSERT_EQ(get_file_size(std::string(index_path_prefix_) + ".data"), 51200008);
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    void *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    uint32_t max_num_threads = 8;
    uint32_t max_search_l = 500;
    uint32_t search_l = 100;
    uint32_t topk = 100;
    load_aligned_bin(data_size_, query_filename_, query, query_num, query_dim, query_aligned_dim);
    load_truthset(gt_filename_, gt_ids, gt_dists, gt_num, gt_dim);
    auto config = IndexConfigBuilder()
                    .with_dimension(query_dim)
                    .with_max_points(0)
                    .is_dynamic_index(false)
                    .is_enable_tags(false)
                    .build();
    size_t num_points = config.max_points + config.num_frozen_pts;
    size_t dim = config.dimension;
    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
            (config.index_write_params == nullptr ? 0 : config.index_write_params->max_degree));
    std::unique_ptr<InMemDataStore> in_mem_data_store_ = std::make_unique<InMemDataStore>(num_points, dim, data_size_);
    in_mem_data_store_->setMethod(_method);
    std::unique_ptr<InMemGraphStore> in_mem_graph_store_ = std::make_unique<InMemGraphStore>(num_points, max_reserve_degree);
    std::unique_ptr<CoarseVamanaIndex> vamana_index = std::make_unique<CoarseVamanaIndex>(data_size_, config, 
                                                        std::move(in_mem_data_store_), std::move(in_mem_graph_store_), false);
    vamana_index->load(index_path_prefix_.c_str(), max_num_threads, max_search_l);
    std::vector<uint64_t> query_result_ids;
    query_result_ids.resize(topk * query_num);
    std::vector<float> query_result_dists;
    query_result_dists.resize(topk * query_num);
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);
    auto s = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
        auto qs = std::chrono::high_resolution_clock::now();
        cmp_stats[i] = vamana_index->search((char *)query + i * query_aligned_dim * data_size_, topk, search_l,
                                            query_result_ids.data() + i * topk, query_result_dists.data() + i * topk).second;
        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[i] = (float)(diff.count() * 1000000);
    }

    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    double displayed_qps = query_num / diff.count();

    std::vector<uint32_t> our_results;
    our_results.resize(topk * query_num);

    for (size_t i = 0; i < our_results.size(); i++) {
        our_results[i] = (uint32_t)query_result_ids[i];
    }

    double recall = calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim, our_results.data(), topk, topk);

    std::sort(latency_stats.begin(), latency_stats.end());
    double mean_latency = std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

    float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

    aligned_free(query);

    LOG_INFO("displayed_qps: %lf; recall: %lf; mean_latency: %lf; avg_cmps: %lf", displayed_qps, recall, mean_latency, avg_cmps);

    ASSERT_GT(recall, 95);
}

TEST_F(CoarseVamanaIndexTest, TestBuildHalf) {
    size_t data_num, data_dim;
    get_bin_metadata(base_filename_, data_num, data_dim);
    auto index_build_params = IndexWriteParametersBuilder(L, R)
                                .with_alpha(alpha)
                                .with_saturate_graph(true)
                                .with_num_threads(num_threads)
                                .build();

    auto config = IndexConfigBuilder()
                    .with_dimension(data_dim)
                    .with_max_points(data_num)
                    .is_dynamic_index(false)
                    .with_index_write_params(index_build_params)
                    .is_enable_tags(false)
                    .build();

    size_t num_points = config.max_points + config.num_frozen_pts;
    size_t dim = config.dimension;
    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
            (config.index_write_params == nullptr ? 0 : config.index_write_params->max_degree));
    std::unique_ptr<InMemDataStore> in_mem_data_store_ = std::make_unique<InMemDataStore>(num_points, dim, half_data_size_);
    in_mem_data_store_->setMethod(_half_method);
    std::unique_ptr<InMemGraphStore> in_mem_graph_store_ = std::make_unique<InMemGraphStore>(num_points, max_reserve_degree);
    std::unique_ptr<CoarseVamanaIndex> vamana_index = std::make_unique<CoarseVamanaIndex>(half_data_size_, config, 
                                                      std::move(in_mem_data_store_), std::move(in_mem_graph_store_), true);

    std::ifstream reader;
    reader.exceptions(std::ios::badbit | std::ios::failbit);
    reader.open(base_filename_.c_str(), std::ios::binary);
    reader.seekg(0, reader.beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (unsigned)npts_i32;

    std::unique_ptr<float[]> data = std::make_unique<float[]>(dim);

    for (size_t i = 0; i < npts; i++)
    {
        reader.read((char *)data.get(), dim * data_size_);
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

        vamana_index->set_data_vec(i, query_info.GetVector());
    }
    vamana_index->build_with_data_populated();
    vamana_index->save(half_index_path_prefix_.c_str());
    vamana_index.reset();
    ASSERT_TRUE(file_exists(half_index_path_prefix_));
    ASSERT_EQ(get_file_size(half_index_path_prefix_), 13200024);
    ASSERT_TRUE(file_exists(std::string(half_index_path_prefix_) + ".data"));
    ASSERT_EQ(get_file_size(std::string(half_index_path_prefix_) + ".data"), 25600008);
}

TEST_F(CoarseVamanaIndexTest, TestSearchHalf) {
    ASSERT_TRUE(file_exists(half_index_path_prefix_));
    ASSERT_EQ(get_file_size(half_index_path_prefix_), 13200024);
    ASSERT_TRUE(file_exists(std::string(half_index_path_prefix_) + ".data"));
    ASSERT_EQ(get_file_size(std::string(half_index_path_prefix_) + ".data"), 25600008);
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    void *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    uint32_t max_num_threads = 8;
    uint32_t max_search_l = 500;
    uint32_t search_l = 100;
    uint32_t topk = 100;
    load_aligned_bin(data_size_, query_filename_, query, query_num, query_dim, query_aligned_dim);
    load_truthset(gt_filename_, gt_ids, gt_dists, gt_num, gt_dim);
    auto config = IndexConfigBuilder()
                    .with_dimension(query_dim)
                    .with_max_points(0)
                    .is_dynamic_index(false)
                    .is_enable_tags(false)
                    .build();
    size_t num_points = config.max_points + config.num_frozen_pts;
    size_t dim = config.dimension;
    size_t max_reserve_degree = (size_t)(defaults::GRAPH_SLACK_FACTOR * 1.05 *
            (config.index_write_params == nullptr ? 0 : config.index_write_params->max_degree));
    std::unique_ptr<InMemDataStore> in_mem_data_store_ = std::make_unique<InMemDataStore>(num_points, dim, half_data_size_);
    in_mem_data_store_->setMethod(_half_method);
    std::unique_ptr<InMemGraphStore> in_mem_graph_store_ = std::make_unique<InMemGraphStore>(num_points, max_reserve_degree);
    std::unique_ptr<CoarseVamanaIndex> vamana_index = std::make_unique<CoarseVamanaIndex>(half_data_size_, config, 
                                                        std::move(in_mem_data_store_), std::move(in_mem_graph_store_), false);
    vamana_index->load(half_index_path_prefix_.c_str(), max_num_threads, max_search_l);
    std::vector<uint64_t> query_result_ids;
    query_result_ids.resize(topk * query_num);
    std::vector<float> query_result_dists;
    query_result_dists.resize(topk * query_num);
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats = std::vector<uint32_t>(query_num, 0);
    auto s = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
        float * cur_query = (float *)((char *)query + i * query_aligned_dim * data_size_);
        std::string search_str;
        for (size_t j = 0; j < query_aligned_dim; j++)
        {
            search_str += std::to_string(cur_query[j]);
            if (j != dim - 1) {
                search_str += " ";
            }
        }
        QueryInfo query_info(search_str);
        
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

        if (!query_info.MakeAsSearcher()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
            return;
        }

        auto qs = std::chrono::high_resolution_clock::now();
        cmp_stats[i] = vamana_index->search(query_info.GetVector(), topk, search_l,
                                            query_result_ids.data() + i * topk,
                                            query_result_dists.data() + i * topk).second;
        auto qe = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = qe - qs;
        latency_stats[i] = (float)(diff.count() * 1000000);
    }

    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
    double displayed_qps = query_num / diff.count();

    std::vector<uint32_t> our_results;
    our_results.resize(topk * query_num);

    for (size_t i = 0; i < our_results.size(); i++) {
        our_results[i] = (uint32_t)query_result_ids[i];
    }

    double recall = calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim, our_results.data(), topk, topk);

    std::sort(latency_stats.begin(), latency_stats.end());
    double mean_latency = std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

    float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

    aligned_free(query);

    LOG_INFO("displayed_qps: %lf; recall: %lf; mean_latency: %lf; avg_cmps: %lf", displayed_qps, recall, mean_latency, avg_cmps);

    ASSERT_GT(recall, 95);
}

MERCURY_NAMESPACE_END(core);
