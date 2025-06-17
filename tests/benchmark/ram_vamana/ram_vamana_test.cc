/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: anduo <anduo@xiaohongshu.com>
/// Created: 2024-06-24 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/algorithm/vamana/ram_vamana_searcher.h"
#include "src/core/algorithm/vamana/ram_vamana_builder.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"
#include "src/core/framework/instance_factory.h"
#include "src/core/utils/string_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

DEFINE_uint32(search_topk, 100, "search topk. default 100");
DEFINE_uint32(thread_num, 15, "thread num. default 15");
DEFINE_uint32(internal_iter, 10, "internal iter. default 10");
DEFINE_string(dataset, "sift", "dataset. choose sift/dssmtopk");

class BenchmarkRamVamanaTest: public testing::Test
{
public:
    void SetUp()
    {
        index_params_.set(PARAM_DATA_TYPE, "half");
        index_params_.set(PARAM_METHOD, "L2");
        index_params_.set(PARAM_DIMENSION, 128);

        index_params_.set(PARAM_INDEX_TYPE, "RamVamana");
        index_params_.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, 64);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, 150);
        index_params_.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, 8);
        index_params_.set(PARAM_GENERAL_MAX_BUILD_NUM, 2000000);


        factory_.SetIndexParams(index_params_);
    }

    void TearDown()
    {
    }

    AlgorithmFactory factory_;
    IndexParams index_params_;
    
    std::string ram_vamana_index_sift_learn_prefix_ = "tests/benchmark/data/ram_index_sift_learn_R32_L50_A1.2";
};

void searchTask(mercury::core::RamVamanaSearcher* searcher, std::vector<QueryInfo>* queryinfos) {
    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);
    for (uint32_t iter = 0; iter < FLAGS_internal_iter; iter++) {
        for (uint32_t i = 0; i < queryinfos->size(); ++i) {
            int ret_code = searcher->Search(queryinfos->at(i), &context);
            context.clean();
        }
    }
    return;
}

TEST_F(BenchmarkRamVamanaTest, TestInit) {
    char *buffer;
    //也可以将buffer作为输出参数
    buffer = getcwd(NULL, 0);
    std::cout << "cwd is:" << buffer << std::endl;

    Builder::Pointer builder_p = factory_.CreateBuilder();
    mercury::core::RamVamanaBuilder* builder = dynamic_cast<mercury::core::RamVamanaBuilder*>(builder_p.get());

    std::string input = std::string("/data1/anduo/mercury/tests/benchmark/data/") + FLAGS_dataset + "_train.csv";
    std::ifstream infile(input.c_str());
    std::string line;
    int docid = 0;
    while(getline(infile,line)) {
        line.pop_back();
        builder->AddDoc(docid++, 0, line);
    }
    infile.close();

    const void *dump_data = nullptr;
    size_t dump_size = 0;
    dump_data = builder->DumpIndex(&dump_size);
    ASSERT_TRUE(dump_data != nullptr);
    std::cout << "dump_size: " << dump_size << std::endl;

    ASSERT_EQ(builder->DumpRamVamanaIndex(ram_vamana_index_sift_learn_prefix_), 0);

    Searcher::Pointer index_searcher = factory_.CreateSearcher();
    mercury::core::RamVamanaSearcher* searcher = dynamic_cast<mercury::core::RamVamanaSearcher *> (index_searcher.get());

    int ret_code = searcher->LoadIndex(dump_data, dump_size);
    ASSERT_EQ(ret_code, 0);
    ret_code = searcher->LoadIndex(ram_vamana_index_sift_learn_prefix_);
    ASSERT_EQ(ret_code, 0);

    IndexParams index_params;
    mercury::core::GeneralSearchContext context(index_params);

    std::vector<QueryInfo> query_infos;
    std::string input_test = std::string("/data1/anduo/mercury/tests/benchmark/data/") + FLAGS_dataset + "_test.csv";
    std::ifstream infile_test(input_test.c_str());
    docid = 0;
    std::string topk = std::to_string(FLAGS_search_topk); 
    while(getline(infile_test,line)) {
        line.pop_back();
        query_infos.resize(docid + 1);
        query_infos[docid].SetQuery(topk + "&0:0#0||" + line);
        query_infos[docid].SetFeatureTypes(IndexMeta::FeatureTypes::kTypeHalfFloat);
        query_infos[docid].MakeAsSearcher();
        docid++;
    }
    infile_test.close();

    // 召回率计算
    std::cout<<"start recall rate"<<std::endl;
    std::string input_neighbor = std::string("/data1/anduo/mercury/tests/benchmark/data/") + FLAGS_dataset + "_neighbors.csv";
    std::ifstream infile_neighbor(input_neighbor.c_str());
    float recallSum = 0;
    uint32_t i = 0;
    while (getline(infile_neighbor,line)) {
        std::set<uint64_t> fngSet;
        std::set<uint64_t> oriSet;

        ret_code = searcher->Search(query_infos[i++], &context);
        for (auto &result : context.Result()) {
            fngSet.insert(result.gloid);
        }
        for (auto &gt : mercury::core::StringUtil::split(line, " ")) {
            uint64_t gt_docid;
            mercury::core::StringUtil::strToUInt64(gt.c_str(), gt_docid);
            oriSet.insert(gt_docid);
        }

        std::set<uint32_t> intersection;
        std::set_intersection(fngSet.begin(), fngSet.end(), 
                              oriSet.begin(), oriSet.end(),
                              std::inserter(intersection, intersection.begin()));

        recallSum += ((float)intersection.size()) / oriSet.size();
        context.clean();
    }
    std::cout<<"recall rate: "<<(recallSum/query_infos.size())<<std::endl;

    std::cout<<"start benchmark"<<std::endl;
    butil::Timer timer;

    // benchmark
    for (uint32_t iter = 0; iter < 10; iter++) {
        fprintf(stdout, "iter: %d start\n", iter);
        timer.start();
        std::vector<std::thread> threads;

        for (int i = 0; i < FLAGS_thread_num; ++i) {
            threads.emplace_back(searchTask, searcher, &query_infos);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        timer.stop();
        uint64_t time = timer.m_elapsed();
        fprintf(stdout, "iter: %d end, topk %d, thread count: %d, time: %lu. qps: %f\n", iter, FLAGS_search_topk, FLAGS_thread_num, time, FLAGS_thread_num*query_infos.size()*FLAGS_internal_iter/(double(time)/1000.0));
    }
}

MERCURY_NAMESPACE_END(core);
