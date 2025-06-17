/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/utils/vamana/in_mem_graph_store.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class InMemGraphStoreTest: public testing::Test
{
public:
    void SetUp()
    {
        std::cout << "InMemGraphStoreTest" << std::endl;
        std::cout << "cwd is:" << getcwd(NULL, 0) << std::endl;
        _total_points = 100000;
        _reserve_graph_degree = 43;
        in_mem_graph_store_ = std::make_unique<InMemGraphStore>(_total_points, _reserve_graph_degree);
    }

    void TearDown()
    {
    }

    size_t _total_points;
    size_t _reserve_graph_degree;
    std::unique_ptr<InMemGraphStore> in_mem_graph_store_;

};

TEST_F(InMemGraphStoreTest, TestGetTotal) {
    std::cout << "TestGetTotal" << std::endl;
    ASSERT_EQ(in_mem_graph_store_->get_total_points(), 100000);
}

MERCURY_NAMESPACE_END(core);
