/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/engine/redindex/redindex_iterator.h"
#undef protected
#undef private

namespace mercury {
namespace redindex {
using namespace mercury::core;

class RedIndexIteratorIvfTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(RedIndexIteratorIvfTest, TestNormalUse)
{
    std::vector<mercury::core::SearchResult> search_results;
    search_results.push_back(mercury::core::SearchResult(0, 0, 1.0));
    search_results.push_back(mercury::core::SearchResult(0, 1, 1.0));
    search_results.push_back(mercury::core::SearchResult(0, 2, 1.0));
    RedIndexIterator iter(std::move(search_results));

    iter.Init(); 
    ASSERT_TRUE(iter.IsValid());
    ASSERT_EQ(0, iter.Data());
    iter.Next();
    ASSERT_TRUE(iter.IsValid());
    ASSERT_EQ(1, iter.Data());
    iter.Next();
    ASSERT_EQ(2, iter.Data());
    ASSERT_TRUE(iter.IsValid());
    iter.Next();
    ASSERT_FALSE(iter.IsValid());
}


}; // namespace redindex
}; // namespace mercury
