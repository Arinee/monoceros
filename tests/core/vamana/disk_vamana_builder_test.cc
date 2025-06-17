/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>

#define protected public
#define private public
#include "src/core/algorithm/vamana/disk_vamana_builder.h"
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class DiskVamanaBuilderTest: public testing::Test
{
public:
    void SetUp()
    {
        index_sift_learn_prefix_ = "tests/core/vamana/test_data/index_sift_learn_R32_L50_A1.2";

        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";

        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);

        index_params_.set(PARAM_INDEX_TYPE, "DiskVamana");
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT, 1000000);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 1000000);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/vamana/test_data/cluster_results");

    }

    void TearDown()
    {

    }

    AlgorithmFactory factory_;
    IndexParams index_params_;

    std::string index_sift_learn_prefix_;
    std::string base_filename_, query_filename_, gt_filename_;
};

TEST_F(DiskVamanaBuilderTest, TestInit) {

    index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, false);
    factory_.SetIndexParams(index_params_);
    
    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::DiskVamanaBuilder* builder = dynamic_cast<mercury::core::DiskVamanaBuilder*>(builder_p.get());
    ASSERT_TRUE(builder != nullptr);
    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(builder->GetIndex().get());
    ASSERT_TRUE(index != nullptr);
    ASSERT_TRUE(index->GetAlignedReader() == nullptr);

    const void* fragmentCenter = index->GetPqCentroidResource().getValueInIntegrateMatrix(31, 255);
    uint32_t size = index->GetPqCentroidResource().getIntegrateMeta().elemSize / 4;
    ASSERT_EQ(4, size);
    EXPECT_NEAR(64.5647f, *((float *)((char *)fragmentCenter + 0 * size)), 0.0001);
    EXPECT_NEAR(80.2931f, *((float *)((char *)fragmentCenter + 1 * size)), 0.0001);
    EXPECT_NEAR(10.6983f, *((float *)((char *)fragmentCenter + 2 * size)), 0.0001);
    EXPECT_NEAR(5.2931f, *((float *)((char *)fragmentCenter + 3 * size)), 0.0001);
}

TEST_F(DiskVamanaBuilderTest, TestParallelProcess) {

    index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, true);
    factory_.SetIndexParams(index_params_);

    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::DiskVamanaBuilder* builder = dynamic_cast<mercury::core::DiskVamanaBuilder*>(builder_p.get());
    ASSERT_TRUE(builder != nullptr);

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(builder->GetIndex().get());
    ASSERT_TRUE(index != nullptr);
    ASSERT_TRUE(index->GetAlignedReader() == nullptr);

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
        for (int k = 0; k < 10; k++) {
            builder->AddDoc(i, 0, build_str);
        }
    }

    ASSERT_EQ(0, builder->BatchProcess());
}

TEST_F(DiskVamanaBuilderTest, TestPartitionBuildAndDump) {

    index_params_.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, true);
    factory_.SetIndexParams(index_params_);

    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::DiskVamanaBuilder* builder = dynamic_cast<mercury::core::DiskVamanaBuilder*>(builder_p.get());
    ASSERT_TRUE(builder != nullptr);

    mercury::core::DiskVamanaIndex* index = dynamic_cast<mercury::core::DiskVamanaIndex*>(builder->GetIndex().get());
    ASSERT_TRUE(index != nullptr);
    ASSERT_TRUE(index->GetAlignedReader() == nullptr);

    auto index_mata = index->GetIndexMeta();

    size_t data_size = index_mata.sizeofElement() / index_mata.dimension();

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
        reader.read((char *)data.get(), dim * data_size);
        std::string build_str;
        for (size_t j = 0; j < dim; j++)
        {
            build_str += std::to_string(data[j]);
            if (j != dim - 1) {
                build_str += " ";
            }
        }
        builder->AddDoc(i, 0, build_str);
    }
    
    std::string vamana_path, pq_path, medoids_path;
    size_t vamana_size, pq_size, medoids_size;
    int ret = builder->MultiPartSingleDump(vamana_path, &vamana_size, pq_path, &pq_size, medoids_path, &medoids_size);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(vamana_path, std::string(get_current_dir_name()) + "/vamana_partition_disk.index");
    ASSERT_EQ(vamana_size, 68272128);
    ASSERT_EQ(pq_path, std::string(get_current_dir_name()) + "/vamana_partition_pq.index");
    ASSERT_EQ(pq_size, 6531648);
    ASSERT_EQ(medoids_path, std::string(get_current_dir_name()) + "/vamana_partition_medoids.bin");
    ASSERT_EQ(medoids_size, 24);
}

MERCURY_NAMESPACE_END(core);
