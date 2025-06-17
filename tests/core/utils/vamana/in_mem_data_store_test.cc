/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/utils/vamana/in_mem_data_store.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class InMemDataStoreTest: public testing::Test
{
public:
    void SetUp()
    {
        std::cout << "InMemDataStoreTest" << std::endl;
        std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        num_base_points_ = 100000;
        dimension_ = 128;
        method_ = IndexDistance::kMethodFloatSquaredEuclidean;
        data_size_ = 4;
        in_mem_data_store_ = std::make_unique<InMemDataStore>(num_base_points_, dimension_, data_size_);
        in_mem_data_store_->setMethod(method_);
    }

    void TearDown()
    {
    }

    std::string base_filename_;
    uint32_t num_base_points_;
    uint16_t data_size_;
    size_t dimension_;
    IndexDistance::Methods method_;
    std::unique_ptr<InMemDataStore> in_mem_data_store_;

};

TEST_F(InMemDataStoreTest, TestLoadFromFile) {
    std::cout << "TestLoadFromFile" << std::endl;
    ASSERT_EQ(in_mem_data_store_->load(base_filename_), 100000);
}

TEST_F(InMemDataStoreTest, TestPopulateData) {
    std::cout << "TestPopulateData" << std::endl;
    size_t file_dim, file_num_points;
    if (!file_exists(base_filename_))
    {
        LOG_ERROR("ERROR: data file %s does not exist.", base_filename_.c_str());
        std::stringstream stream;
        stream << "ERROR: data file " << base_filename_ << " does not exist." << std::endl;
        throw new std::runtime_error(stream.str());
    }

    get_bin_metadata(base_filename_, file_num_points, file_dim);

    if (file_dim != 128)
    {
        LOG_ERROR("ERROR: Driver requests loading %lu dimension, but file has %lu dimension.", file_dim, file_dim);
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << 128 << " dimension,"
            << "but file has " << file_dim << " dimension." << std::endl;
        throw new std::runtime_error(stream.str());
    }

    in_mem_data_store_->populate_data(base_filename_, 0U);
    float *_aligned_query = nullptr;
    alloc_aligned(((void **)&_aligned_query), 128 * sizeof(float), 8 * sizeof(float));
    // get the first float of the first vector
    in_mem_data_store_->get_vector(0, _aligned_query);
    ASSERT_EQ(_aligned_query[0], 97);
}

TEST_F(InMemDataStoreTest, TestGetVector) {
    std::cout << "TestGetVector" << std::endl;
    in_mem_data_store_->load(base_filename_);
    float *_aligned_query = nullptr;
    alloc_aligned(((void **)&_aligned_query), 128 * sizeof(float), 8 * sizeof(float));
    // get the last float of the last vector
    in_mem_data_store_->get_vector(99999, _aligned_query);
    ASSERT_EQ(_aligned_query[127], 1);
}

TEST_F(InMemDataStoreTest, TestCalculateDistance) {
    std::cout << "TestCalculateDistance" << std::endl;
    ASSERT_EQ(in_mem_data_store_->_method, IndexDistance::kMethodFloatSquaredEuclidean);
    in_mem_data_store_->load(base_filename_);
    float *_aligned_query = nullptr;
    alloc_aligned(((void **)&_aligned_query), 128 * sizeof(float), 8 * sizeof(float));
    in_mem_data_store_->get_vector(0, _aligned_query);
    float dist1 = in_mem_data_store_->get_distance(_aligned_query, 99999);
    float dist2 = in_mem_data_store_->get_distance(_aligned_query, 0);
    ASSERT_EQ(dist1, 198153);
    ASSERT_EQ(dist2, 0);
}

MERCURY_NAMESPACE_END(core);
