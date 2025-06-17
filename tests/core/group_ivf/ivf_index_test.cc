/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-23 10:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
//#include "src/core/framework/index_storage.h"
//#include "src/core/framework/instance_factory.h"
//#include "src/core/utils/string_util.h"
#include "src/core/algorithm/ivf/ivf_index.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class IvfIndexTest: public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 0.5);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data/");
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 64);
        index_params_.set(PARAM_INDEX_TYPE, "Ivf");
        factory_.SetIndexParams(index_params_);
    }

    void TearDown()
    {
    }

    AlgorithmFactory factory_;
    IndexParams index_params_;
};


TEST_F(IvfIndexTest, TestMemQuota)
{
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;
    Index::Pointer index = factory_.CreateIndex();
    mercury::core::IvfIndex* core_index = dynamic_cast<mercury::core::IvfIndex*>(index.get());

    ASSERT_TRUE(core_index != nullptr);

    EXPECT_EQ(64, core_index->index_meta_.dimension());
    EXPECT_EQ(129800000, core_index->GetMaxDocNum());
    size_t memQuota = 536870912;
    size_t elemSize = 256 * sizeof(float);
    std::cout << "---------- test memQuota: " << memQuota << std::endl;
    std::cout << "---------- test elemSize: " << elemSize << std::endl;
    EXPECT_EQ(62800000, core_index->MemQuota2DocCount(memQuota, elemSize));
    memQuota = 100000000;
}

MERCURY_NAMESPACE_END(core);
