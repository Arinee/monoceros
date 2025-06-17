/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-06 00:59

#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
// #include <sys/types.h>
// #include <sys/stat.h>
// #include <unistd.h>

#define protected public
#define private public
#include "common.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/group_ivf/group_ivf_merger.h"
#include "src/core/algorithm/group_ivf/group_ivf_searcher.h"
#include "src/core/algorithm/group_ivf/mocked_vector_reader.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/algorithm/thread_common.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/utils/note_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupIvfSearcherMipsTest : public testing::Test
{
public:
    void SetUp()
    {
        char *buffer;
        buffer = getcwd(NULL, 0);
        std::cout << "cwd is:" << buffer << std::endl;
        index_params_.set(PARAM_COARSE_SCAN_RATIO, 1);
        index_params_.set(PARAM_TRAIN_DATA_PATH, "tests/core/group_ivf/test_data_mips/");
        index_params_.set(PARAM_DATA_TYPE, "float");
        index_params_.set(PARAM_METHOD, "IP");
        index_params_.set(PARAM_DIMENSION, 64);
        index_params_.set(PARAM_ENABLE_MIPS, true);
        index_params_.set(PARAM_ENABLE_FORCE_HALF, true);
        index_params_.set(PARAM_INDEX_TYPE, "GroupIvf");
        index_params_.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        factory_.SetIndexParams(index_params_);
    }

    void TearDown() {}

    AlgorithmFactory factory_;
    IndexParams index_params_;
};

TEST_F(GroupIvfSearcherMipsTest, TestNormSearchInMips)
{
    char *buffer;
    // 也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Index::Pointer index1 = factory_.CreateIndex();
    mercury::core::GroupIvfIndex *core_index = dynamic_cast<mercury::core::GroupIvfIndex *>(index1.get());
    gindex_t group_index = 0;
    SlotIndex label = 0;
    docid_t docid = 0;
    std::string group_str =
        "-33.90547 -59.706287 48.835728 -62.356342 33.936466 48.50636 -26.751385 59.75746 33.03566 45.579685 "
        "-48.838688 46.006645 -58.90832 55.893734 -41.172283 -47.753666 49.472294 44.115376 49.11006 -51.995327 "
        "47.064056 -47.813168 57.25176 -51.85131 -54.570953 -11.404261 42.909225 37.42667 55.335953 -49.228226 "
        "41.26257 28.407585 -46.533516 -71.79979 -51.861336 51.693302 41.67177 -63.30279 31.241753 -55.64836 68.10441 "
        "63.062897 40.105938 -76.04086 -42.56255 -36.57254 62.094345 -47.054977 -62.917274 -2.0498137 -54.17733 "
        "-51.361908 -53.814766 -56.469204 34.480488 -40.896263 2.9311256 39.31956 -50.87228 53.057777 43.05973 "
        "-64.21518 -42.149433 58.226173";

    int ret = core_index->Add(docid++, INVALID_PK, group_str, "abcdefghigklmnop");
    ASSERT_EQ(0, ret);

    group_str =
        "-34.002544 -59.708866 48.670128 -62.263477 33.880093 48.678436 -26.732231 59.670372 33.187496 45.659855 "
        "-48.743027 45.92701 -58.9042 55.99621 -41.334087 -47.98175 49.65195 44.082558 49.006645 -51.927856 47.047295 "
        "-48.013496 57.11601 -51.79472 -54.89465 -11.264078 42.9839 37.449326 55.230564 -49.198864 41.289196 28.328003 "
        "-46.633503 -71.87357 -51.707954 51.667213 41.76134 -63.42013 31.099493 -55.3945 68.021286 63.18376 40.157124 "
        "-76.03606 -42.647327 -36.561443 62.008293 -47.390587 -63.07466 -2.1193652 -54.255424 -51.248665 -53.89301 "
        "-56.645004 34.418564 -40.6987 2.869772 39.232693 -50.872715 53.036346 43.021366 -64.13377 -42.1961 58.391636";
    ret = core_index->Add(docid++, INVALID_PK, group_str, "abcdefghigklmnop");
    ASSERT_EQ(0, ret);

    ASSERT_EQ(2, core_index->GetDocNum());

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    size_t dump_size;
    const void *dump_data = nullptr;
    ret = core_index->Dump(dump_data, dump_size);
    ASSERT_EQ(0, ret);
    ASSERT_TRUE(dump_data != nullptr);
    ret = index_searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(0, ret);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    std::string search_str = "0:0#100||" + group_str;
    QueryInfo query_info(search_str);
    ASSERT_TRUE(query_info.MakeAsSearcher());
    ret = index_searcher->Search(query_info, &context);

    ASSERT_EQ(ret, 0);
    ASSERT_EQ(2, context.Result().size());
    ASSERT_EQ(0, context.Result().at(0).gloid);
    ASSERT_FLOAT_EQ(-155693.84, context.Result().at(0).score);
    ASSERT_EQ(1, context.Result().at(1).gloid);
    ASSERT_FLOAT_EQ(-155732.38, context.Result().at(1).score);
}

MERCURY_NAMESPACE_END(core);
