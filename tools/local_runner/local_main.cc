#include <gflags/gflags.h>
#include <iostream>

#include "BaseFileReader.h"
#include "QueryFileReader.h"

#include "alog/Logger.h"
#include "alog/Configurator.h"

#include "src/core/framework/instance_factory.h"
#include "src/core/algorithm/algorithm_factory.h"
#include "src/core/algorithm/group_hnsw/group_hnsw_builder.h"
#include "src/core/algorithm/group_hnsw/group_hnsw_searcher.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/framework/index_storage.h"

using namespace mercury::core;

DEFINE_string(log_conf_file, "", "The log config file path(required)");
DEFINE_string(base_file, "", "The base doc file to build ANNS index (required)");
DEFINE_string(query_file, "", "The request query file for ANNS index search (required)");
DEFINE_string(result_file, "", "The result file path(required)");

int main(int argc, char* argv[]) {

    LOG_INFO("Start Monoceros Local Runner");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_log_conf_file.empty()) {
        std::cerr << "Error: --log_conf_file is required but not provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " --log_conf_file=<path>" << std::endl;
        std::exit(1);
    }

    std::cout << "log config file path is " << FLAGS_log_conf_file << std::endl;

    alog::Configurator::configureLogger(FLAGS_log_conf_file.c_str());

    if (FLAGS_base_file.empty()) {
        std::cerr << "Error: --base_file is required but not provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " --base_file=<path>" << std::endl;
        std::exit(1);
    }

    LOG_INFO("The path of base file is %s", FLAGS_base_file.c_str());

    if (FLAGS_query_file.empty()) {
        std::cerr << "Error: --query_file is required but not provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " --query_file=<path>" << std::endl;
        std::exit(1);
    }

    LOG_INFO("The path of query file is %s", FLAGS_query_file.c_str());

    if (FLAGS_result_file.empty()) {
        std::cerr << "Error: --result_file is required but not provided." << std::endl;
        std::cerr << "Usage: " << argv[0] << " --result_file=<path>" << std::endl;
        std::exit(1);
    }

    LOG_INFO("The path of result file is %s", FLAGS_result_file.c_str());

    IndexParams index_params;

    index_params.set(PARAM_DATA_TYPE, "half");
    index_params.set(PARAM_METHOD, "L2");
    index_params.set(PARAM_DIMENSION, 64);
    index_params.set(PARAM_GROUP_HNSW_BUILD_THRESHOLD, 5);
    index_params.set(PARAM_HNSW_BUILDER_MAX_LEVEL, 10);
    index_params.set(PARAM_HNSW_BUILDER_SCALING_FACTOR, 30);
    index_params.set(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT, 20);
    index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, 10200);
    index_params.set(PARAM_GRAPH_COMMON_SEARCH_STEP, 10);
    index_params.set(PARAM_HNSW_BUILDER_EFCONSTRUCTION, 400);
    index_params.set(PARAM_GRAPH_COMMON_MAX_SCAN_NUM, 10000);
    index_params.set(PARAM_INDEX_TYPE, "GroupHnsw");
    index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);

    AlgorithmFactory factory;

    factory.SetIndexParams(index_params);

    Builder::Pointer builder_p = factory.CreateBuilder();

    mercury::core::GroupHnswBuilder* builder = dynamic_cast<mercury::core::GroupHnswBuilder*>(builder_p.get());

    if (builder == nullptr) {
        std::cerr << "Error: failed to init group hnsw builder!" << std::endl;
        exit(2);
    }

    BaseFileReader baseFileReader;
    std::map<int32_t, std::pair<std::string, std::string>> id_pk_vec_holder = baseFileReader.readFileAndProcessLines(FLAGS_base_file);

    for (const auto& entry : id_pk_vec_holder) {
        int ret_code = builder->AddDoc(entry.first, 0, entry.second.second);
        if (ret_code != 0) {
            std::cerr << "Error: failed to add doc to index!" << std::endl;
            exit(3);
        }
    }

    const void *dump_content = nullptr;
    size_t dump_size = 0;
    dump_content = builder->DumpIndex(&dump_size);
    if (dump_content == nullptr) {
        std::cerr << "Error: index dump error with null content!" << std::endl;
        exit(4);
    }

    std::cout << "dump size is " << dump_size << std::endl;

    Searcher::Pointer index_searcher = factory.CreateSearcher();
    mercury::core::GroupHnswSearcher* searcher = dynamic_cast<mercury::core::GroupHnswSearcher *> (index_searcher.get());
    if (searcher == nullptr) {
        std::cerr << "Error: failed to create searcher!" << std::endl;
        exit(5);
    }

    int ret_code = searcher->LoadIndex(dump_content, dump_size);
    if (ret_code != 0) {
        std::cerr << "Error: failed to load index!" << std::endl;
        exit(6);
    }

    if (searcher->getFType() != IndexMeta::kTypeHalfFloat) {
        std::cerr << "Error: data type mismatch!" << std::endl;
        exit(7);
    }

    QueryFileReader reader(FLAGS_query_file);

    std::vector<std::pair<std::string, std::string>> requests = reader.getRequests();

    IndexParams index_params_search;
    mercury::core::GeneralSearchContext context(index_params_search);

    std::ofstream outFile(FLAGS_result_file);

    for (const auto& pair : requests) {
        outFile << "request_id: " << pair.first << std::endl;
        outFile << "request_url: " << pair.second << std::endl;
        QueryInfo query_info(pair.second);
        query_info.SetFeatureTypes(mercury::core::IndexMeta::kTypeHalfFloat);
        if (!query_info.MakeAsSearcher()) {
            std::cerr << "Error: failed to make as searcher!" << std::endl;
            exit(8);
        }
        int ret_code = searcher->Search(query_info, &context);
        if (ret_code != 0) {
            std::cerr << "Error: search failed!" << std::endl;
            exit(8);
        }
        outFile << "return topk: " << context.Result().size() << std::endl;
        outFile << "return results:" << std::endl;
        for (SearchResult result : context.Result()) {
            outFile << "id=" << id_pk_vec_holder.at(result.gloid).first << ";" << "score=" << result.score << std::endl;
        }
        context.clean();
    }

    outFile.close();

    alog::Logger::shutdown();

    return 0;
}
