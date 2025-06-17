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

class DiskVamanaHalfSingleIndexTest: public testing::Test
{
public:
    void SetUp()
    {
        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";
        mem_index_sift_learn_prefix_ = "tests/core/vamana/test_data/disk_in_mem_half_single_index_sift_learn_R32_L50_A1.2";
        disk_index_sift_learn_prefix_ = "tests/core/vamana/test_data/disk_in_SSD_half_single_index_sift_learn_R32_L50_A1.2";

        size_t data_num, data_dim;
        get_bin_metadata(base_filename_, data_num, data_dim);

        sift_index_params_.set(PARAM_DATA_TYPE, "half");
        sift_index_params_.set(PARAM_METHOD, "L2");
        sift_index_params_.set(PARAM_DIMENSION, data_dim);

        sift_index_params_.set(PARAM_INDEX_TYPE, "DiskVamana");
        sift_index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_ALPHA, 2.0);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION, 3000);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED, false);
        sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        sift_index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, data_num);
        sift_index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/vamana/test_data/cluster_results");
    }

    void TearDown()
    {

    }

    AlgorithmFactory factory_, sift_factory_;
    IndexParams index_params_, sift_index_params_;

    std::string base_filename_, query_filename_, gt_filename_, mem_index_sift_learn_prefix_, disk_index_sift_learn_prefix_;
};

TEST_F(DiskVamanaHalfSingleIndexTest, TestBuildAndDump) {

    sift_factory_.SetIndexParams(sift_index_params_);

    Index::Pointer index_p = sift_factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

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

        index->BaseIndexAdd(i, 0, query_info.GetVector(), query_info.GetVectorLen());
    }

    index->BuildMemIndex();

    index->DumpMemLocal(mem_index_sift_learn_prefix_);

    index->CreateDiskLayout(mem_index_sift_learn_prefix_ + ".data", mem_index_sift_learn_prefix_, disk_index_sift_learn_prefix_);

    ASSERT_TRUE(file_exists(disk_index_sift_learn_prefix_));

    ASSERT_EQ(get_file_size(disk_index_sift_learn_prefix_), 40964096);
}

TEST_F(DiskVamanaHalfSingleIndexTest, TestLoadAndSearch) {

    sift_factory_.SetIndexParams(sift_index_params_);

    Index::Pointer index_p = sift_factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    auto index_mata = index->GetIndexMeta();

    size_t ori_data_size = 4;

    index->BuildPqIndexFromFile(base_filename_);

    ASSERT_TRUE(file_exists(disk_index_sift_learn_prefix_));

    ASSERT_EQ(get_file_size(disk_index_sift_learn_prefix_), 40964096);

    index->LoadDiskIndex(disk_index_sift_learn_prefix_, "");

    std::vector<uint32_t> node_list;

    uint32_t num_nodes_to_cache = 10000;

    std::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;

    index->cache_bfs_levels(num_nodes_to_cache, node_list);

    index->load_cache_list(node_list);

    node_list.clear();
    
    node_list.shrink_to_fit();

    // load query bin && truthset
    void *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    load_aligned_bin(ori_data_size, query_filename_, query, query_num, query_dim, query_aligned_dim);
    load_truthset(gt_filename_, gt_ids, gt_dists, gt_num, gt_dim);
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);

    const uint32_t recall_at = 100;

    uint32_t L = 140;

    uint32_t optimized_beamwidth = 2;

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(17) << "99.9 Latency" << std::setw(15) << "Mean IO" 
                  << std::setw(16) << "Mean CPU" << std::setw(17)  << "Mean ios" << std::setw(16) << "Mean cmps" 
                  << std::setw(16) << "Mean hops" << std::setw(16)  << recall_string << std::endl;
    std::cout << "=================================================================================="
                 "==================================================================================" << std::endl;
    std::vector<uint32_t> query_result_id(recall_at * query_num);
    std::vector<float> query_result_dist(recall_at * query_num);
    auto stats = new QueryStats[query_num];
    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    omp_set_num_threads(4);
    auto s = std::chrono::high_resolution_clock::now();
    query_num = 100;
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)query_num; i++) {
        float * cur_query = (float *)((char *)query + i * query_aligned_dim * ori_data_size);
        std::string search_str;
        for (size_t j = 0; j < query_aligned_dim; j++)
        {
            search_str += std::to_string(cur_query[j]);
            if (j != query_dim - 1) {
                search_str += " ";
            }
        }
        QueryInfo query_info(search_str);
        
        query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

        if (!query_info.MakeAsSearcher()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
        }

        QueryInfo query_info_raw(query_info.GetRawQuery());
        
        if (!query_info_raw.MakeAsSearcher()) {
            LOG_ERROR("resolve query failed. query str:%s", query_info_raw.GetRawQuery().c_str());
        }

        index->cached_beam_search_half(query_info.GetVector(), query_info_raw.GetVector(), recall_at, L,
                                    query_result_ids_64.data() + (i * recall_at),
                                    query_result_dist.data() + (i * recall_at),
                                    optimized_beamwidth, stats + i);
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    double qps = (1.0 * query_num) / (1.0 * diff.count());

    convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_id.data(), query_num, recall_at);
    
    auto mean_latency = get_mean_stats<float>(stats, query_num, [](const QueryStats &stats) { return stats.total_us; });

    auto latency_999 = get_percentile_stats<float>(stats, query_num, 0.999, [](const QueryStats &stats) { return stats.total_us; });

    auto mean_ios = get_mean_stats<uint32_t>(stats, query_num, [](const QueryStats &stats) { return stats.io_us; });

    auto mean_cpuus = get_mean_stats<float>(stats, query_num, [](const QueryStats &stats) { return stats.cpu_us; });

    auto mean_n_ios = get_mean_stats<uint32_t>(stats, query_num, [](const QueryStats &stats) { return stats.n_ios; });

    auto mean_n_cmps = get_mean_stats<uint32_t>(stats, query_num, [](const QueryStats &stats) { return stats.n_cmps; });

    auto mean_n_hops = get_mean_stats<uint32_t>(stats, query_num, [](const QueryStats &stats) { return stats.n_hops; });

    double recall = calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim, query_result_id.data(), recall_at, recall_at);

    std::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus << std::setw(16) << mean_n_ios << std::setw(16) << mean_n_cmps
                      << std::setw(16) << mean_n_hops<< std::setw(16) << recall << std::endl;
    delete[] stats;

    aligned_free(query);

    index_p.reset();
}

MERCURY_NAMESPACE_END(core);
