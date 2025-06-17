#include "bench_result.h"
#include "txt_input_reader.h"
#include "common/params_define.h"
#include "utils/string_util.h"
#include "gflags/gflags.h"
#include "framework/index_framework.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using namespace mercury;
using namespace std;

DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(service_class, "IvfpqService", "The register name of service");
DEFINE_string(index, ".", "The dir of output indexes");
DEFINE_string(query, "query", "The query file");
DEFINE_int32(max_iter, 100000, "max iter num of query");
DEFINE_int32(topk, 50, "topk return number");
DEFINE_int32(threads, 50, "Thread number");
DEFINE_string(pq_rough, "0.05,0.01", "Coarse scan ratio");
DEFINE_int32(pq_integrate, 500, "Product scan number");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");

template <typename T>
class Bench
{
public:
    Bench(int threads) 
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

    bool loadQuery(const std::string &queryFile, 
                   const std::string &firstSep,
                   const std::string &secondSep)
    {
        cout << "Loading query..." << endl;
        TxtInputReader<T> reader;
        return reader.loadQuery(queryFile, firstSep, secondSep, _queries);
    }

    void run(VectorService::Pointer service, 
             int maxIter,
             int topk)
    {
        // Check
        if (_queries.size() == 0) {
            LOG_INFO("Query size is 0");
            return;
        }

        // Do bench
        _benchResult.markStart();
        for (int i = 0; i < maxIter; ++i) {
            int idx = i % _queries.size();
            Closure::Pointer task = Closure::New(
                    this, 
                    &Bench::startBench,
                    service,
                    _queries[idx],
                    topk);
            _pool->enqueue(task, true);
        }
        _pool->waitFinish();
        _benchResult.markEnd();
        _benchResult.print();
    }

private:

    void startBench(VectorService::Pointer service,
                    const vector<T> &query,
                    int topk)
    {
        struct timeval start, end;
        gettimeofday(&start, nullptr);

        // Prepare shared context
        IndexParams params;
        VectorService::SearchContext::Pointer context = 
            service->CreateContext(params);
        if (!context) {
            cerr << "Failed to create search context" << endl;
            return;
        }

        // Do Search
        int ret = service->KnnSearch(topk, query.data(), query.size()*sizeof(float), context);
        if (ret != 0) {
            cerr << "Failed to Search, ret=" << ret << endl;
            return;
        }

        gettimeofday(&end, nullptr);
        
        // Check result
        auto &result = context->Result();
        if (result.empty()) {
            cerr << "Search result is empty" << endl;
        }

        // Do sample
        int64_t processTimeUs = 
            (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        _benchResult.addTime(processTimeUs);
    }

private:
    size_t _threads;
    shared_ptr<ThreadPool> _pool;
    vector<vector<T>> _queries;
    BenchResult _benchResult;
};

int main(int argc, char * argv[]) 
{
    //gflags
    gflags::SetUsageMessage("Usage: bench [options]");
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

    // Create a service
    VectorService::Pointer service =
        InstanceFactory::CreateService(FLAGS_service_class.c_str());
    if (!service) {
        cerr << "Failed to create " << FLAGS_service_class << endl;
        return -1;
    } 
    cout << "Created " << FLAGS_service_class << endl;

    //Prepare params
    IndexParams params;
    params.set(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO, FLAGS_pq_rough);
    params.set(PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM, FLAGS_pq_integrate);

    // Init the service
    int ret = service->Init(params);
    if (ret != 0) {
        cerr << "Failed to init service." << endl;
        return -1;
    }

    // Load Index using Storage
    IndexStorage::Pointer storage =
        InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!storage) {
        cerr << "Failed to create storage " << FLAGS_storage_class << endl;
        return -1;
    } 
    cout << "Created storage " << FLAGS_storage_class << endl;

    // Load Index NOW
    ret = service->LoadIndex(FLAGS_index, storage);
    if (ret < 0) {
        cerr << "Failed to load Index, index dir: " << FLAGS_index  << endl;
        return -1;
    }
    cout << "Load Index done!" << endl;

    string firstSep = FLAGS_input_first_sep;
    string secondSep = FLAGS_input_second_sep;
    // Start bench
    if (FLAGS_type == "float") {
        Bench<float> bench(FLAGS_threads);
        bench.loadQuery(FLAGS_query, firstSep, secondSep);
        bench.run(service, FLAGS_max_iter, FLAGS_topk);
    } else if (FLAGS_type == "int16") {
        Bench<int16_t> bench(FLAGS_threads);
        bench.loadQuery(FLAGS_query, firstSep, secondSep);
        bench.run(service, FLAGS_max_iter, FLAGS_topk);
    } else if (FLAGS_type == "int8") {
        Bench<int8_t> bench(FLAGS_threads);
        bench.loadQuery(FLAGS_query, firstSep, secondSep);
        bench.run(service, FLAGS_max_iter, FLAGS_topk);
    } else if (FLAGS_type == "binary") {
        Bench<uint32_t> bench(FLAGS_threads);
        bench.loadQuery(FLAGS_query, firstSep, secondSep);
        bench.run(service, FLAGS_max_iter, FLAGS_topk);
    } else {
        cerr << "Can not recognize type: " << FLAGS_type << endl;
    }
    cout << "Bench done." << endl;

    return 0;
}

