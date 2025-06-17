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

class DiskVamanaHalfIndexTest: public testing::Test
{
public:
    void SetUp()
    {
        // test using random mock data:
        mem_index_mock_prefix_ = "tests/core/vamana/test_data/disk_in_mem_half_index_mock_R32_L50_A1.2";
        disk_index_mock_prefix_ = "tests/core/vamana/test_data/disk_in_SSD_half_index_mock_R32_L50_A1.2";

        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);

        index_params_.set(PARAM_INDEX_TYPE, "DiskVamana");
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 10000);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, false);

        factory_.SetIndexParams(index_params_);

        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";
        mem_index_sift_learn_prefix_ = "tests/core/vamana/test_data/disk_in_mem_half_index_sift_learn_R32_L50_A1.2";
        disk_index_sift_learn_prefix_ = "tests/core/vamana/test_data/disk_in_SSD_half_index_sift_learn_R32_L50_A1.2";

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

    std::string mem_index_mock_prefix_, disk_index_mock_prefix_;
    std::string base_filename_, query_filename_, gt_filename_, mem_index_sift_learn_prefix_, disk_index_sift_learn_prefix_;
};

TEST_F(DiskVamanaHalfIndexTest, TestBuildVamanaMem) {

    GTEST_SKIP() << "Skipping DiskVamanaHalfIndexTest.TestBuildVamanaMem";

    Index::Pointer index_p = factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    std::string build_str1 = "-0.042992 -0.047437 0.016824 -0.032322 -0.058978 0.00603 0.004624 -0.102875 0.043804 -0.129003 -0.063305 -0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.115877 -0.054619 0.11011 0.096558 0.018475 -0.087097 0.026629 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034821 -0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 -0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 -0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 -0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 -0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 -0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467";

    QueryInfo query_info1(build_str1);

    query_info1.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

    if (!query_info1.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info1.GetRawQuery().c_str());
        return;
    }

    std::string build_str2 = "-0.102875 -0.047437 0.016824 -0.032322 -0.018908 0.00603 0.004624 -0.042992 0.073804 -0.129003 0.05032171308994293 -0.018899 0.047719 0.03726 -0.012384 0.024918 0.063164 0.003078 -0.054183 -0.118877 -0.054619 0.11011 0.096558 0.018475 -0.087097 0.026679 -0.08566 0.02428 -0.088073 -0.046569 -0.094956 -0.027088 0.034621 -0.106306 -0.008979 0.031298 0.030987 0.123223 -0.004162 -0.04761 -0.037318 0.039261 0.136555 0.019844 -0.042915 0.100461 -0.061183 0.028209 -0.084324 0.012819 0.028181 -0.08822 0.035264 -0.072143 -0.015044 0.111965 -0.031478 0.068231 -0.018672 0.001338 0.046578 0.067762 -0.057287 -0.047343 -0.005129 0.161173 -0.021491 -0.003186 0.015636 0.006977 0.003917 -0.031233 0.024885 0.071695 -0.05812 -0.099307 -0.061015 0.031348 0.074765 -0.005988 -0.036811 0.062878 -0.117832 -0.013208 0.037202 0.017278 0.132848 -0.028692 0.107519 0.04048 -0.07074 -0.038141 -0.116665 -0.049325 0.138709 0.02446 -0.006483 0.097473 0.019068 -0.030756 0.003464 -0.004049 -0.017734 -0.049131 0.011849 -0.087313 0.018132 0.093301 0.02402 -0.114743 -0.073973 0.022781 -0.003511 0.043055 0.069776 -0.050155 -0.018691 0.078315 -0.016702 0.042535 -0.088182 -0.055301 0.038858 0.051842 0.048028 -0.044823 -0.051279 -0.016467";

    QueryInfo query_info2(build_str2);

    query_info2.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

    if (!query_info2.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info2.GetRawQuery().c_str());
        return;
    }

    std::string build_str3 = "0.2677592635154724 0.28430742025375366 0.1048450618982315 0.05032171308994293 -0.06191530451178551 0.03726 0.035384856164455414 -0.08079756051301956 -0.03292011469602585 -0.07846521586179733 0.17667590081691742 -0.07388703525066376 -0.22193367779254913 -0.09453566372394562 0.033058810979127884 -0.03693600744009018 0.0069535477086901665 -0.03590690344572067 -0.10681653022766113 -0.005090558901429176 0.19453756511211395 -0.09202316403388977 0.008568838238716125 0.16556806862354279 0.0893283411860466 0.011802629567682743 -0.15821677446365356 -0.035604771226644516 0.07552704960107803 0.38303399085998535 0.23279926180839539 0.2613544762134552 0.08599590510129929 -0.16477899253368378 -0.13135594129562378 0.054203152656555176 0.03629051148891449 -0.1589570939540863 0.03334221616387367 -0.09955812990665436 0.09106072038412094 0.10482850670814514 -0.16112633049488068 -0.08260127156972885 -0.06246046721935272 0.1025809645652771 0.19028225541114807 -0.008005745708942413 0.036354463547468185 0.004253556951880455 0.02039208635687828 0.19594474136829376 0.14263570308685303 0.022878360003232956 0.12011625617742538 0.07643178850412369 -0.015086587518453598 0.13617363572120667 -0.018525924533605576 -0.005557427182793617 -0.1378810703754425 0.0569249652326107 0.3010271489620209 0.08654335141181946 0.2677592635154724 0.28430742025375366 0.1048450618982315 0.05032171308994293 -0.06191530451178551 0.12526169419288635 0.035384856164455414 -0.08079756051301956 -0.03292011469602585 -0.07846521586179733 0.17667590081691742 -0.07388703525066376 -0.22193367779254913 -0.09453566372394562 0.033058810979127884 -0.03693600744009018 0.0069535477086901665 -0.03590690344572067 -0.10681653022766113 -0.005090558901429176 0.19453756511211395 -0.09202316403388977 0.008568838238716125 0.16556806862354279 0.0893283411860466 0.011802629567682743 -0.15821677446365356 -0.035604771226644516 0.07552704960107803 0.38303399085998535 0.23279926180839539 0.2613544762134552 0.08599590510129929 -0.16477899253368378 -0.13135594129562378 0.054203152656555176 0.03629051148891449 -0.1589570939540863 0.03334221616387367 -0.09955812990665436 0.09106072038412094 0.10482850670814514 -0.16112633049488068 -0.08260127156972885 -0.06246046721935272 0.1025809645652771 0.19028225541114807 -0.008005745708942413 0.036354463547468185 0.004253556951880455 0.02039208635687828 0.19594474136829376 0.14263570308685303 0.022878360003232956 0.12011625617742538 0.07643178850412369 -0.015086587518453598 0.13617363572120667 -0.018525924533605576 -0.005557427182793617 -0.1378810703754425 0.0569249652326107 0.3010271489620209 0.08654335141181946";

    QueryInfo query_info3(build_str3);

    query_info3.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);

    if (!query_info3.MakeAsBuilder()) {
        LOG_ERROR("resolve query failed. query str:%s", query_info3.GetRawQuery().c_str());
        return;
    }

    for (int i = 0; i < 2000; i++) {
        index->BaseIndexAdd(i, 0, query_info1.GetVector(), query_info1.GetVectorLen());
    }

    for (int i = 2000; i < 5000; i++) {
        index->BaseIndexAdd(i, 0, query_info2.GetVector(), query_info2.GetVectorLen());
    }

    for (int i = 5000; i < 10000; i++) {
        index->BaseIndexAdd(i, 0, query_info3.GetVector(), query_info3.GetVectorLen());
    }

    index->BuildMemIndex();

    index->DumpMemLocal(mem_index_mock_prefix_);

    ASSERT_TRUE(file_exists(mem_index_mock_prefix_));

    ASSERT_TRUE(file_exists(mem_index_mock_prefix_ + ".data"));

    ASSERT_EQ(get_file_size(mem_index_mock_prefix_), 1320024);

    ASSERT_EQ(get_file_size(mem_index_mock_prefix_ + ".data"), 2560008);

}

TEST_F(DiskVamanaHalfIndexTest, TestCreateDiskLayout) {

    GTEST_SKIP() << "Skipping DiskVamanaHalfIndexTest.TestCreateDiskLayout";

    Index::Pointer index_p = factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    ASSERT_TRUE(file_exists(mem_index_mock_prefix_));

    ASSERT_TRUE(file_exists(mem_index_mock_prefix_ + ".data"));

    ASSERT_EQ(get_file_size(mem_index_mock_prefix_), 1320024);

    ASSERT_EQ(get_file_size(mem_index_mock_prefix_ + ".data"), 2560008);

    index->CreateDiskLayout(mem_index_mock_prefix_ + ".data", mem_index_mock_prefix_, disk_index_mock_prefix_);

    ASSERT_TRUE(file_exists(disk_index_mock_prefix_));

    ASSERT_EQ(get_file_size(disk_index_mock_prefix_), 4100096);

}

TEST_F(DiskVamanaHalfIndexTest, TestSiftLearnPartitionBuildAndMerge) {

    sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, true);

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

        QueryInfo query_info_raw(query_info.GetRawQuery());

        if (!query_info_raw.MakeAsBuilder()) {
            LOG_ERROR("resolve query_info_raw failed. query_info_raw str:%s", query_info_raw.GetRawQuery().c_str());
            return;
        }

        index->PartitionBaseIndexAdd(i, 0, query_info, query_info_raw);
    }

    index->PartitionBaseIndexDump();

    index->BuildAndDumpPartitionIndex();

    std::string path_medoids;

    size_t size_medoids;
        
    index->MergePartitionIndex(path_medoids, &size_medoids);

    index->CreateDiskLayout(std::string(get_current_dir_name()) + "/vamana_partition_half_ori.data", 
                            std::string(get_current_dir_name()) + "/vamana_partition_half_merged_mem.index", 
                            std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index");

    ASSERT_TRUE(file_exists(std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index"));

    ASSERT_EQ(get_file_size(std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index"), 40964096);

    ASSERT_TRUE(file_exists(path_medoids));

    ASSERT_EQ(size_medoids, 24);
}

TEST_F(DiskVamanaHalfIndexTest, TestBuildAndMergeWithCusDup) {

    sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, true);

    sift_index_params_.set(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR, 3);

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

        QueryInfo query_info_raw(query_info.GetRawQuery());

        if (!query_info_raw.MakeAsBuilder()) {
            LOG_ERROR("resolve query_info_raw failed. query_info_raw str:%s", query_info_raw.GetRawQuery().c_str());
            return;
        }

        index->PartitionBaseIndexAdd(i, 0, query_info, query_info_raw);
    }

    index->PartitionBaseIndexDump();

    index->BuildAndDumpPartitionIndex();

    std::string path_medoids;

    size_t size_medoids;
        
    index->MergePartitionIndex(path_medoids, &size_medoids);

    index->CreateDiskLayout(std::string(get_current_dir_name()) + "/vamana_partition_half_ori.data", 
                            std::string(get_current_dir_name()) + "/vamana_partition_half_merged_mem.index", 
                            std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index");

    ASSERT_TRUE(file_exists(std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index"));

    ASSERT_EQ(get_file_size(std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index"), 40964096);

    ASSERT_TRUE(file_exists(path_medoids));

    ASSERT_EQ(size_medoids, 24);
}

TEST_F(DiskVamanaHalfIndexTest, TestSiftLearnLoadAndSearchPartitionStats) {

    sift_factory_.SetIndexParams(sift_index_params_);

    Index::Pointer index_p = sift_factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    auto index_mata = index->GetIndexMeta();

    size_t ori_data_size = 4;

    std::string ori_data_path = std::string(get_current_dir_name()) + "/vamana_partition_half_raw.data";

    ASSERT_TRUE(file_exists(ori_data_path));

    index->BuildPqIndexFromFile(ori_data_path);

    std::string disk_index_path = std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index";

    ASSERT_TRUE(file_exists(disk_index_path));

    std::string medoids_data_path = std::string(get_current_dir_name()) + "/vamana_partition_half_medoids.bin";

    ASSERT_TRUE(file_exists(medoids_data_path));

    index->LoadDiskIndex(disk_index_path, medoids_data_path);

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

TEST_F(DiskVamanaHalfIndexTest, TestSiftLearnLoadCache) {

    sift_factory_.SetIndexParams(sift_index_params_);

    Index::Pointer index_p = sift_factory_.CreateIndex();

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(index_p.get());

    ASSERT_TRUE(index != nullptr);

    auto index_mata = index->GetIndexMeta();

    std::string disk_index_path = std::string(get_current_dir_name()) + "/vamana_partition_half_disk.index";

    ASSERT_TRUE(file_exists(disk_index_path));

    std::string medoids_data_path = std::string(get_current_dir_name()) + "/vamana_partition_half_medoids.bin";

    ASSERT_TRUE(file_exists(medoids_data_path));

    index->LoadDiskIndex(disk_index_path, medoids_data_path);

    std::vector<uint32_t> node_list;

    uint32_t num_nodes_to_cache = 1000;

    std::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;

    index->cache_bfs_levels(num_nodes_to_cache, node_list);

    std::cout << "size of node_list is " << node_list.size() << std::endl;;

    index->load_cache_list(node_list);

    node_list.clear();
    
    node_list.shrink_to_fit();

    index_p.reset();
}

MERCURY_NAMESPACE_END(core);
