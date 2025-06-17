/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <numeric>

#define protected public
#define private public
#include "src/core/algorithm/vamana/ram_vamana_builder.h"
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class RamVamanaBuilderTest: public testing::Test
{
public:
    void SetUp()
    {
        ram_index_sift_learn_prefix_ = "tests/core/vamana/test_data/ram_index_sift_learn_R32_L50_A1.2";

        // test using sift_learn data:
        base_filename_ = "tests/core/vamana/test_data/sift_learn.fbin";
        query_filename_ = "tests/core/vamana/test_data/sift_query.fbin";
        gt_filename_ = "tests/core/vamana/test_data/sift_query_learn_gt100";

        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);

        index_params_.set(PARAM_INDEX_TYPE, "RamVamana");
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 100000);
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 32);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 50);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);

    }

    void TearDown()
    {

    }

    AlgorithmFactory factory_;
    IndexParams index_params_;

    std::string ram_index_sift_learn_prefix_;
    std::string base_filename_, query_filename_, gt_filename_;
};

TEST_F(RamVamanaBuilderTest, TestBuildAndDump) {

    factory_.SetIndexParams(index_params_);
    
    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::RamVamanaBuilder* builder = dynamic_cast<mercury::core::RamVamanaBuilder*>(builder_p.get());
    ASSERT_TRUE(builder != nullptr);
    mercury::core::RamVamanaIndex* index = dynamic_cast<mercury::core::RamVamanaIndex*>(builder->GetIndex().get());
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
        builder->AddDoc(i, 0, build_str);
    }

    const void *dump_content = nullptr;

    size_t dump_size = 0;

    dump_content = builder->DumpIndex(&dump_size);

    ASSERT_EQ(dump_size, 544);

    ASSERT_TRUE(dump_content != nullptr);

    builder->DumpRamVamanaIndex(ram_index_sift_learn_prefix_);

    ASSERT_TRUE(file_exists(ram_index_sift_learn_prefix_));

    ASSERT_TRUE(file_exists(ram_index_sift_learn_prefix_ + ".data"));

    ASSERT_EQ(get_file_size(ram_index_sift_learn_prefix_), 13200024);
}

MERCURY_NAMESPACE_END(core);
