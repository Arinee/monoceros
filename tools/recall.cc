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
DEFINE_string(topk, "1,10,50", "Mutlti recall value");
DEFINE_uint64(threads, 0, "Thread number");
DEFINE_string(gt, "", "The groud truth file");
DEFINE_string(pq_rough, "0.05,0.01", "Coarse scan ratio");
DEFINE_uint64(pq_integrate, 500, "Product scan number");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_int32(search_method, 0, "specify search-used distance method, default using builder method");

mutex recall_lock;

template <typename T>
class Recall
{
public:
    // default using groudtruth
    Recall(size_t threads, const string &output, bool gtMode = true) 
        : _threads(threads)
        , _output(output)
        , _gtMode(gtMode)
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
             const string& recallTops, 
             const string& gtFile)
    {
        if (! loadGT(gtFile)) {
            cerr << "Load ground truth file failed!" << endl;
            return;
        }

        if (_gtMode && _gt.size() != _queries.size()) {
            cerr << "ground truth size[" << _gt.size() 
                 << "] is not match query size[" << _queries.size() << "]" << endl;
            return;
        }

        if (_queries.size() < _threads) {
            _threads = _queries.size();
            _pool = make_shared<ThreadPool>(true, _threads);
            LOG_DEBUG("Resize thread pool count[%lu]", _threads);
        }

        // Try to mkdir 
        string cmd = "mkdir -p " + _output;
        system(cmd.c_str());

        // Prepare file handler
        vector<pair<fstream*, fstream*>> outputFS;
        struct stat sb;
        if (stat(_output.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
            cout << "logs output to : " << FLAGS_output << endl;
            for (size_t i = 0; i < _threads; ++i) {
                fstream *fsK = new fstream();
                fsK->open(_output + "/t" + to_string(i) + ".knn", ios::out);
                fstream *fsL = new fstream();
                fsL->open(_output + "/t" + to_string(i) + ".linear", ios::out);
                outputFS.push_back(make_pair(fsK, fsL));
            }
        }

        vector<size_t> recallValues;
        StringUtil::fromString(recallTops, recallValues, ",");
        for (auto i : recallValues) {
            _recallRes[i] = 0.0f;
        }
        size_t topk = _recallRes.rbegin()->first;

        if (_gtMode && topk > _gt[0].size()) {
            cerr << "Ground truth node size is smaller than topk[" << topk << "]" << endl;
            return;
        }
        for (size_t i = 0; i < _queries.size(); ++i) {
            Closure::Pointer task = Closure::New(
                    this, 
                    &Recall::recallOne, 
                    service, 
                    _queries[i], 
                    topk, 
                    i, 
                    outputFS);
            _pool->enqueue(task, true);
        }
        _pool->waitFinish();

        for (auto fs : outputFS) {
            fs.first->close();
            fs.second->close();
            delete fs.first;
            delete fs.second;
        }

        for (auto it : _recallRes) {
            cout << "Recall@" << it.first << ": " << it.second / _queries.size() << endl; 
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

    bool loadGT(const string& gtFile)
    {
        if (gtFile.empty()) {
            _gtMode = false;
            return true;
        }

        ifstream gtf(gtFile, ios::binary);
        if (! gtf.is_open()) {
            cerr << "Open ground truth file failed! [" << gtFile << "]" << endl;
            return false;
        }
        gtf.seekg(0, ios::end);
        size_t fileSize = gtf.tellg();
        gtf.seekg(0, ios::beg);

        const size_t LENGTH = 10240;
        size_t gap = sizeof(int);
        size_t size = sizeof(uint64_t) + sizeof(float);
        char *buffer = new char[LENGTH];
        gtf.read(buffer, gap);
        size_t D = (size_t)*(int*)buffer;
        size_t line = gap + size * D;
        size_t N = fileSize / line;
        if (line > LENGTH) {
            delete[] buffer;
            buffer = new char[line];
        }

        gtf.seekg(0, ios::beg);
        for (size_t n = 0; n < N; ++n) {
            gtf.read(buffer, line);
            vector<pair<uint64_t, float>> oneGT;
            oneGT.reserve(D);

            for (size_t i = 0; i < D; ++i) {
                uint64_t key = *(uint64_t*)(buffer + gap + size * i);
                float score = *(float*)(buffer + gap + size * i + sizeof(uint64_t));
                oneGT.emplace_back(key, score);
            }
            _gt.emplace_back(oneGT);
        }

        return true;
    }

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

    void recallOne(VectorService::Pointer service, 
                   vector<T>& query, size_t topk, size_t idx, 
                   vector<pair<fstream*, fstream*>>& outputFS)
    {
        size_t threadIndex = _pool->getThisIndex();
        fstream* knnFS = nullptr;
        fstream* linearFS = nullptr;
        if (outputFS.size() > threadIndex) {
            knnFS = outputFS[threadIndex].first;
            linearFS = outputFS[threadIndex].second;
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
        
        vector<SearchResult> linearRes;
        if (! _gtMode) {
            VectorService::SearchContext::Pointer linearContext = 
                service->CreateContext(params);
            if (!knnContext || !linearContext) {
                cerr << "Failed to create search context" << endl;
                return;
            }
            ret = service->ExhaustiveSearch(topk, query.data(), query.size()*sizeof(T), linearContext);
            if (ret < 0) {
                cerr << "Failed to linearSearch, ret=" << ret << endl;
                return ;
            }
            linearRes = linearContext->Result();
        } else {
            for (size_t i = 0; i < topk; ++i) {
                auto gtNode = _gt[idx][i];
                linearRes.emplace_back(gtNode.first, gtNode.first, gtNode.second);
            }
        }

        if (knnFS) {
            for (auto knn : knnRes) {
                string str = "query[" + to_string(idx) + "]\tkey[" + 
                             to_string(knn.key) + "], dist[" + 
                             to_string(knn.score) + "]\n";
                knnFS->write(str.c_str(), str.size());
            }
        }
        size_t match = 0;
        for (size_t i = 0, j = 0; i < linearRes.size() && j < knnRes.size(); ) {
            bool m = false;
            if (fabs(linearRes[i].score - knnRes[j].score) < 0.000001) {
                ++j;
                match++;
                m = true;
            } else if (linearRes[i].score > knnRes[j].score) {
                cerr << "Linear result is greater than knn result, query key: " << idx << endl;
                fprintf(stderr, "linear result: index[%lu], score[%f], key[%lu]\n", i, linearRes[i].score, linearRes[i].key);
                fprintf(stderr, "knn result: index[%lu], score[%f], key[%lu]\n", j, knnRes[j].score, knnRes[j].key);
                cerr << "query: ";
                for (size_t k = 0; k < query.size(); ++k) {
                    cerr << query[k] << " ";
                }
                cerr << endl;
                return;
            }

            if (linearFS) {
                string str = string(m ? "    HIT" : "NOT HIT") + "  query[" + 
                             to_string(idx) + "]\tkey[" + to_string(linearRes[i].key) + 
                             "], dist[" + to_string(linearRes[i].score) + "]\n";
                linearFS->write(str.c_str(), str.size());
            }

            ++i;
            auto it = _recallRes.find(i);
            if (it != _recallRes.end()) {
                lock_guard<mutex> lock(recall_lock);
                it->second += 100.0 * match / i ;
            }
        }
    }
    
private:
    size_t _threads;
    string _output;
    shared_ptr<ThreadPool> _pool;
    vector<vector<T>> _queries;
    map<size_t, float> _recallRes;
    bool _gtMode;
    vector<vector<pair<uint64_t, float>>> _gt;
};

int main(int argc, char *argv[])
{
    //gflags
    gflags::SetUsageMessage("Usage: recall [options]");
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

    // Calculate Recall
    string firstSep = FLAGS_input_first_sep;
    string secondSep = FLAGS_input_second_sep;
    if (FLAGS_type == "float") {
        Recall<float> recall(FLAGS_threads, FLAGS_output);
        recall.loadQuery(FLAGS_query, firstSep, secondSep);
        recall.run(service, FLAGS_topk, FLAGS_gt);
    } else if (FLAGS_type == "int16") {
        Recall<int16_t> recall(FLAGS_threads, FLAGS_output);
        recall.loadQuery(FLAGS_query, firstSep, secondSep);
        recall.run(service, FLAGS_topk, FLAGS_gt);
    } else if (FLAGS_type == "int8") {
        Recall<int8_t> recall(FLAGS_threads, FLAGS_output);
        recall.loadQuery(FLAGS_query, firstSep, secondSep);
        recall.run(service, FLAGS_topk, FLAGS_gt);
    } else if (FLAGS_type == "binary") {
        Recall<uint32_t> recall(FLAGS_threads, FLAGS_output);
        recall.loadQuery(FLAGS_query, firstSep, secondSep);
        recall.run(service, FLAGS_topk, FLAGS_gt);
    } else {
        cerr << "Can not recognize type: " << FLAGS_type << endl;
    }
    cout << "Recall done." << endl;

    // Cleanup
    service->UnloadIndex();
    service->Cleanup();
    return 0;
}
