#include <iostream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include "framework/index_framework.h"
#include "framework/utility/thread_pool.h"
#include "gflags/gflags.h"
#include "utils/string_util.h"
#include "common/params_define.h"
#include "common/txt_file_holder.h"
#include "utils/index_meta_helper.h"
#include "txt_input_reader.h"

using namespace std;
using namespace mercury;

DEFINE_string(service_class, "IvfpqService", "The register name of service");
DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(index, ".", "The dir of output indexes");
DEFINE_string(query, "query", "The query file");
DEFINE_string(output, "groundtruth.ivecs", "The output file");
DEFINE_string(input_first_sep, " ", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_int32(topk, 50, "topk");
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

    void run(VectorService::Pointer service, 
            const string& queryFile, 
            const string& outputFile, 
            int topk)
    {
        // Load Query
        TxtInputReader<T> reader;
        if (! reader.loadQuery(queryFile, 
                    FLAGS_input_first_sep,
                    FLAGS_input_second_sep,
                    _queries)) {
            cerr << "Load querys failed!" << endl;
            return;
        }

        // Resize gt
        _gt.resize(_queries.size());

        for (size_t i = 0; i < _queries.size(); ++i) {
            Closure::Pointer task = Closure::New(
                    this, 
                    &GroundTruth::runOne, 
                    service, 
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
        for (size_t i = 0; i < _gt.size(); ++i) {
            fwrite(&topk, sizeof(int), 1, wfp);
            for (size_t t = 0; t < _gt[i].size(); ++t) {
                fwrite(&(_gt[i][t].first), sizeof(uint64_t), 1, wfp);
                fwrite(&(_gt[i][t].second), sizeof(float), 1, wfp);
            }
        }
        fclose(wfp);
    }


    void runOne(VectorService::Pointer service, 
            vector<T>& query, size_t topk, size_t idx)
    {
        IndexParams params;
        VectorService::SearchContext::Pointer 
            context = service->CreateContext(params);
        if (!context) {
            cerr << "Failed to create search context" << endl;
            return;
        }

        int ret = service->ExhaustiveSearch(topk, query.data(), query.size()*sizeof(float), context);
        if (ret < 0) {
            cerr << "Failed to exhaustive Search, ret=" << ret << endl;
            return;
        }
        auto result = context->Result();
        _gt[idx].resize(result.size());
        for (size_t t = 0; t < result.size(); ++t) {
            _gt[idx][t].first = result[t].key;
            _gt[idx][t].second = result[t].score;
        }
    }

private:
    vector<vector<T>> _queries;
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
    VectorService::Pointer service =
        InstanceFactory::CreateService(FLAGS_service_class);
    if (!service) {
        cerr << "Failed to create " << FLAGS_service_class << endl;
        return -1;
    } 

    // Init the service
    IndexParams params; 
    int iRet = service->Init(params);
    if (iRet < 0) {
        cerr << "Failed to init seacher, ret=" << iRet << endl;
        return iRet;
    }

    // Load Index NOW
    iRet = service->LoadIndex(FLAGS_index, storage);
    if (iRet < 0) {
        cerr << "Failed to loadIndex, ret=" << iRet << endl;
        return iRet;
    }
    cout << "Load Index done!" << endl;

    // Do search
    if (FLAGS_type == "float") {
        GroundTruth<float> groundTruth(FLAGS_threads);
        groundTruth.run(service, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "int16") {
        GroundTruth<int16_t> groundTruth(FLAGS_threads);
        groundTruth.run(service, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "int8") {
        GroundTruth<int8_t> groundTruth(FLAGS_threads);
        groundTruth.run(service, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else if (FLAGS_type == "binary") {
        GroundTruth<uint32_t> groundTruth(FLAGS_threads);
        groundTruth.run(service, FLAGS_query, FLAGS_output, FLAGS_topk);
    } else {
        cerr << "Can not recognize type: " << FLAGS_type << endl;
        return -1;
    }

    // Cleanup
    service->UnloadIndex();
    service->Cleanup();

    return 0;
}
