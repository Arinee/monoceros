#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include "src/core/framework/index_framework.h"
#include "src/core/framework/utility/thread_pool.h"
#include "gflags/gflags.h"
#include "src/core/utils/string_util.h"
//#include "common/params_define.h"
//#include "utils/index_meta_helper.h"
#include "txt_string_reader.h"
#include "src/core/algorithm/group_ivf/group_ivf_searcher.h"
#include "src/core/algorithm/algorithm_factory.h"

using namespace std;
using namespace mercury::core;
using namespace mercury;

DEFINE_string(service_class, "GroupIvf", "The register name of service");
DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(index, ".", "The dir of output indexes");
DEFINE_string(query, "query", "The query file");
DEFINE_string(output, "groundtruth.ivecs", "The output file");
DEFINE_string(input_first_sep, " ", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_int32(topk, 100, "topk");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_uint64(threads, 0, "Thread number");

template <typename T>
class GroundTruth
{
public:
    GroundTruth(size_t threads)
        : _threads(threads)
    {
        if (_threads == 0) {
            _pool = make_shared<ThreadPool>(true);
            _threads = _pool->count();
            LOG_DEBUG("Using cpu count as thread pool count[%lu]", _threads);
        } else {
            _pool = make_shared<ThreadPool>(true, _threads);
            LOG_DEBUG("Using thread pool count[%lu]", _threads);
        }
    }

    void run(Searcher::Pointer searcher,
            const string& queryFile, 
            const string& outputFile, 
            int topk)
    {
        // Load Query
        TxtStringReader<T> reader;
        if (! reader.loadQuery(queryFile, 
                    FLAGS_input_first_sep,
                    _queries)) {
            cerr << "Load querys failed!" << endl;
            return;
        }

        // Resize gt
        //std::cout<< "query size:" << _queries.size() << std::endl;
        _gt.resize(_queries.size());
        //std::cout<< "gt size:" << _gt.size() << std::endl;

        for (size_t i = 0; i < _queries.size(); ++i) {
            Closure::Pointer task = Closure::New(
                    this, 
                    &GroundTruth::runOne, 
                    searcher,
                    _queries[i], 
                    topk, 
                    i);
            _pool->enqueue(task, true);
        }
        _pool->waitFinish();

        FILE *wfp = fopen(outputFile.c_str(), "wb");
        if (!wfp) {
            cerr << "Fail to open: " << outputFile << endl;
            return;
        }

        _gt.resize(_queries.size());
        std::cout<< "gt size:" << _gt.size() << std::endl;
        for (size_t i = 0; i < _gt.size(); ++i) {
            fwrite(&topk, sizeof(int), 1, wfp);
            for (size_t t = 0; t < _gt[i].size(); ++t) {
                fwrite(&(_gt[i][t].first), sizeof(uint64_t), 1, wfp);
                fwrite(&(_gt[i][t].second), sizeof(float), 1, wfp);
            }
        }
        fclose(wfp);
    }


    void runOne(Searcher::Pointer searcher,
                const std::string& query, size_t topk, size_t idx)
    {
        IndexParams params;
        GeneralSearchContext context(params);
        int ret = searcher->Search(query, &context);
        if (ret < 0) {
            cerr << "Failed to exhaustive Search, ret=" << ret << endl;
            return;
        }
        auto result = context.Result();
        std::sort(result.begin(), result.end(), [](SearchResult& a, SearchResult& b) {
                                                    return a.score < b.score;
                                                });
        if (result.size() != 100) {
            std::cout<<result.size()<<std::endl;
            std::cout<<query<<std::endl;
        }

        _gt[idx].resize(result.size());
        for (size_t t = 0; t < result.size(); ++t) {
            _gt[idx][t].first = result[t].key;
            _gt[idx][t].second = result[t].score;
        }
    }

private:
    std::vector<std::string> _queries;
    shared_ptr<ThreadPool> _pool;
    vector<vector<pair<uint64_t, float>>> _gt;
    size_t _threads;
};

int main(int argc, char *argv[])
{
    //gflags
    gflags::SetUsageMessage("Usage: make_groundtruth [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load plugins first
    IndexPluginBroker broker;
    for (int i = 1; i < argc; ++i) {
        const char *file_path = argv[i];
        if (!broker.emplace(file_path)) {
            cerr << "Failed to load plugin " << file_path << endl;
        } else {
            cout << "Loaded plugin " << file_path << endl;
        }
    }

    // Load Index using Storage
    IndexStorage::Pointer storage =
        InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!storage) {
        cerr << "Failed to create storage " << FLAGS_storage_class
                  << endl;
        return 3;
    } 
    cout << "Created storage " << FLAGS_storage_class << endl;

    // Create a Service
    AlgorithmFactory factory;
    IndexParams index_params;
    index_params.set(PARAM_COARSE_SCAN_RATIO, 1);
    index_params.set(PARAM_DATA_TYPE, "float");
    index_params.set(PARAM_METHOD, "L2");
    //index_params.set(PARAM_DIMENSION, 256);
    index_params.set(PARAM_INDEX_TYPE, FLAGS_service_class);
    factory.SetIndexParams(index_params);
    Searcher::Pointer searcher = factory.CreateSearcher();

    if (!searcher) {
        cerr << "Failed to create " << FLAGS_service_class << endl;
        return -1;
    } 

    // Load Index NOW
    IndexStorage::Handler::Pointer handler = storage->open(FLAGS_index, false);
    void *data = nullptr;
    handler->read((const void **)(&data), handler->size());
    int iRet = searcher->LoadIndex(data, handler->size());
    if (iRet != 0) {
        cerr << "Failed to loadIndex, ret=" << iRet << endl;
        return iRet;
    }
    cout << "Load Index done!" << endl;

    // Do search
    if (FLAGS_type == "float") {
        GroundTruth<float> groundTruth(FLAGS_threads);
        groundTruth.run(searcher, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "int16") {
        GroundTruth<int16_t> groundTruth(FLAGS_threads);
        groundTruth.run(searcher, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "int8") {
        GroundTruth<int8_t> groundTruth(FLAGS_threads);
        groundTruth.run(searcher, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "binary") {
        GroundTruth<uint32_t> groundTruth(FLAGS_threads);
        groundTruth.run(searcher, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else {
        cerr << "Can not recognize type: " << FLAGS_type << endl;
        return -1;
    }

    // Cleanup
    //service->UnloadIndex();
    //service->Cleanup();

    return 0;
}
