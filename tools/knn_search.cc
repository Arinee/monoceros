#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include "framework/index_framework.h"
#include "framework/utility/thread_pool.h"
#include "gflags/gflags.h"
#include "utils/string_util.h"
#include "common/params_define.h"
#include "txt_input_reader.h"
#include "framework/search_result.h"

using namespace std;
using namespace mercury;

DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(service_class, "IvfpqService", "The register name of service");
DEFINE_string(output, "output", "The logs output directory");
DEFINE_string(index, ".", "The dir of output indexes");
DEFINE_string(query, "query", "The query file");
DEFINE_uint32(topk, 50, "return topk value");
DEFINE_uint64(threads, 0, "Thread number");
DEFINE_string(pq_rough, "0.05,0.01", "Coarse scan ratio");
DEFINE_uint64(pq_integrate, 500, "Product scan number");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_int32(search_method, 0, "specify search-used distance method, default using builder method");


template <typename T>
class KnnSearch
{
public:
    // default using groudtruth
    KnnSearch(size_t threads, const string &output) 
        : _threads(threads)
        , _output(output)
    {
        if (_threads == 0) {
            _pool = make_shared<ThreadPool>(false);
            _threads = _pool->count();
            LOG_DEBUG("Using cpu count as thread pool count[%lu]", _threads);
        } else {
            _pool = make_shared<ThreadPool>(false, _threads);
            LOG_DEBUG("Using thread pool count[%lu]", _threads);
        }
    }

    void run(VectorService::Pointer service,
             size_t topk)
    {

        if (_queries.size() < _threads) {
            _threads = _queries.size();
            _pool = make_shared<ThreadPool>(true, _threads);
            LOG_DEBUG("Resize thread pool count[%lu]", _threads);
        }

        // Try to mkdir 
        string cmd = "mkdir -p " + _output;
        system(cmd.c_str());

        // Prepare file handler
        vector<fstream *> outputFS;
        struct stat sb;
        if (stat(_output.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            cout << "logs output to : " << FLAGS_output << endl;
            for (size_t i = 0; i < _threads; ++i) {
                fstream *fsK = new fstream();
                fsK->open(_output + "/t" + to_string(i) + ".knn", ios::out);
                outputFS.push_back(fsK);
            }
        }

        for (size_t i = 0; i < _queries.size(); ++i) {
            Closure::Pointer task = Closure::New(
                    this, 
                    &KnnSearch::oneQuerySearch,
                    service, 
                    _queries[i], 
                    topk, 
                    i, 
                    outputFS);
            _pool->enqueue(task, true);
        }
        _pool->waitFinish();

        for (auto fs : outputFS) {
            fs->close();
            delete fs;
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


private:

    template<typename U>
    typename std::enable_if<std::is_same<int16_t, U>::value || 
                            std::is_same<float, U>::value || 
                            std::is_same<int8_t, U>::value, int>::type
    doKnnSearch(VectorService::Pointer service,
            VectorService::SearchContext::Pointer& context,
            const vector<U> &query,
            size_t topk)
    {
        // Do knnSearch
        return service->KnnSearch(topk, query.data(), query.size(), context);
    }

    template<typename U>
    typename std::enable_if<std::is_same<uint32_t, U>::value, int>::type
    doKnnSearch(VectorService::Pointer service,
            VectorService::SearchContext::Pointer& context,
            const vector<U> &query,
            size_t topk)
    {
        // Do linearSearch
        return service->KnnSearch(topk, query.data(), query.size()*32, context);
    }

    void oneQuerySearch(VectorService::Pointer service,
                   vector<T>& query, size_t topk, size_t idx, 
                   vector<fstream*>& outputFS)
    {
        size_t threadIndex = _pool->getThisIndex();
        fstream* knnFS = nullptr;
        if (outputFS.size() > threadIndex) {
            knnFS = outputFS[threadIndex];
        }
        
        IndexParams params;
        VectorService::SearchContext::Pointer 
            knnContext = service->CreateContext(params);
        if (!knnContext) {
            cerr << "Failed to create search context" << endl;
            return;
        }
    
        int ret = service->KnnSearch(topk, query.data(), query.size()*sizeof(T), knnContext);
        if (ret < 0) {
            cerr << "Failed to knnSearch, ret=" << ret << endl;
            return ;
        }
    
        auto knnRes = knnContext->Result();

        if (knnFS) {
            for (auto knn : knnRes) {
                string str = "query[" + to_string(idx) + "]\tkey[" + 
                             to_string(knn.key) + "], dist[" + 
                             to_string(knn.score) + "]\n";
                knnFS->write(str.c_str(), str.size());
            }
        }
    }
    
private:
    size_t _threads;
    string _output;
    shared_ptr<ThreadPool> _pool;
    vector<vector<T>> _queries;
};

int main(int argc, char *argv[])
{
    //gflags
    gflags::SetUsageMessage("Usage: knn_search [options]");
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

    // Create a vector service
    VectorService::Pointer service =
        InstanceFactory::CreateService(FLAGS_service_class.c_str());
    if (!service) {
        cerr << "Failed to create " << FLAGS_service_class
                  << endl;
        return 1;
    } else {
        cout << "Created " << FLAGS_service_class << endl;
    }
 
    // Init the service
    IndexParams params; 
    params.set(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO, FLAGS_pq_rough);
    params.set(PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM, FLAGS_pq_integrate);
    params.set(PARAM_GENERAL_SEARCHER_SEARCH_METHOD, 
            static_cast<IndexDistance::Methods>(FLAGS_search_method));
    int ret = service->Init(params);
    if (ret < 0) {
        cerr << "Failed to init seacher, ret=" << ret << endl;
        return 2;
    }

    // Load Index using Storage
    IndexStorage::Pointer stg =
        InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!stg) {
        cerr << "Failed to create storage " << FLAGS_storage_class
                  << endl;
        return 3;
    } else {
        cout << "Created storage " << FLAGS_storage_class << endl;
    }

    // Load Index NOW
    ret = service->LoadIndex(FLAGS_index, stg);
    if (ret < 0) {
        cerr << "Failed to loadIndex, ret=" << ret << endl;
        return 4;
    }
    cout << "Load Index done!" << endl;

    // Calculate KnnSearch
    string firstSep = FLAGS_input_first_sep;
    string secondSep = FLAGS_input_second_sep;
    if (FLAGS_type == "float") {
        KnnSearch<float> knnSearch(FLAGS_threads, FLAGS_output);
        knnSearch.loadQuery(FLAGS_query, firstSep, secondSep);
        knnSearch.run(service, FLAGS_topk);
    } else if (FLAGS_type == "int16") {
        KnnSearch<int16_t> knnSearch(FLAGS_threads, FLAGS_output);
        knnSearch.loadQuery(FLAGS_query, firstSep, secondSep);
        knnSearch.run(service, FLAGS_topk);
    } else if (FLAGS_type == "int8") {
        KnnSearch<int8_t> knnSearch(FLAGS_threads, FLAGS_output);
        knnSearch.loadQuery(FLAGS_query, firstSep, secondSep);
        knnSearch.run(service, FLAGS_topk);
    } else if (FLAGS_type == "binary") {
        KnnSearch<uint32_t> knnSearch(FLAGS_threads, FLAGS_output);
        knnSearch.loadQuery(FLAGS_query, firstSep, secondSep);
        knnSearch.run(service, FLAGS_topk);
    } else {
        cerr << "Can not recognize type: " << FLAGS_type << endl;
    }
    cout << "KnnSearch done." << endl;

    // Cleanup
    service->UnloadIndex();
    service->Cleanup();
    return 0;
}
