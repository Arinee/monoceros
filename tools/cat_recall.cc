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
#include "index/general_search_context.h"
#include <algorithm>
#include <thread>
#include <unordered_map>

using namespace std;
using namespace mercury;

template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
        // unless the result is subnormal
        || std::abs(x-y) < std::numeric_limits<T>::min();
}

DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(flat_service_class, "CatFlatService", "The flat register name of service");
DEFINE_string(ivf_service_class, "CatIvfflatService", "The ivf register name of service");
DEFINE_string(output, "output", "The logs output directory");
DEFINE_string(flat_index, "cat_flat.indexes", "The dir of output indexes");
DEFINE_string(ivf_index, "cat_ivfflat.indexes", "The dir of output indexes");
DEFINE_string(query, "query", "The query file");
DEFINE_uint64(topk, 10, "top K");
DEFINE_uint64(threads, 10, "Thread number");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, "|", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");

const char DELIM_0 = '|';
const char DELIM_1 = ',';
const size_t SEARCH_LIMIT = 500000;

void Tokenize(const std::string& str_, const char delim_, std::vector<std::string>& res_) {
    size_t start = 0, end = 0;
    while ((start = str_.find_first_not_of(delim_, end)) != std::string::npos) {
        end = str_.find(delim_, start);
        res_.push_back(str_.substr(start, end - start));
    }
}

bool LoadQuery(std::vector<std::pair<std::vector<cat_t>, std::vector<float>>>& query_) {
    std::cout << "loading query..." << std::endl;
    std::ifstream sIn(FLAGS_query);
    if (!sIn) {
        std::cerr << "Failed to open file: " << FLAGS_query << std::endl;
        return false;
    }

    std::string lineStr;
    std::vector<std::string> lineVec;
    while(std::getline(sIn,lineStr)) {
        Tokenize(lineStr, DELIM_0, lineVec);
        if (lineVec.size () != 2) {
            lineStr.clear();
            lineVec.clear();
            continue;
            std::cerr << "LoadQuery:Unexpected line: " << lineStr << std::endl;
            return false;
        }
        std::vector<std::string> catVec;
        std::vector<std::string> dataVec;
        Tokenize(lineVec[0], DELIM_1, catVec);
        Tokenize(lineVec[1], DELIM_1, dataVec);

        std::vector<cat_t> cat;
        std::vector<float> data;
        for (const auto& e : catVec) cat.push_back(std::stoul(e));
        for (const auto& e : dataVec) data.push_back(std::stof(e));
        query_.push_back(std::make_pair(cat, data));

        lineStr.clear();
        lineVec.clear();
    }

    std::cout << "loaded query size: " << query_.size() << std::endl;

    return true;
}

bool LoadPlugin(int argc, char *argv[]) {
    IndexPluginBroker broker;
    for (int i = 1; i < argc; ++i) {
        const char *file_path = argv[i];
        if (!broker.emplace(file_path)) cerr << "Failed to load plugin " << file_path << endl;
        else cout << "Loaded plugin " << file_path << endl;
    }
    return true;
}

bool CreateService(VectorService::Pointer& serviceFlat_,
                    VectorService::Pointer& serviceIvf_) {
    serviceFlat_ = InstanceFactory::CreateService(FLAGS_flat_service_class.c_str());
    serviceIvf_ = InstanceFactory::CreateService(FLAGS_ivf_service_class.c_str());
    if (!serviceFlat_ || !serviceIvf_) {
        std::cerr << "Failed to create service: " << FLAGS_flat_service_class <<  " and " << FLAGS_ivf_service_class << std::endl;
        return false;
    }
    std::cout << FLAGS_flat_service_class << " and " << FLAGS_ivf_service_class << " is created." << std::endl;
    return true;
}

bool InitService(VectorService::Pointer& service) {
    IndexParams params; 
    params.set(PARAM_GENERAL_SEARCHER_SEARCH_METHOD, 
            //mercury::IndexDistance::kMethodFloatSquaredEuclidean);
            mercury::IndexDistance::kMethodFloatInnerProduct);
    int ret = service->Init(params);
    if (ret < 0) {
        cerr << "Failed to init seacher, ret=" << ret << endl;
        return false;
    }
    return true;
}

bool LoadFlatIndex(VectorService::Pointer& service) {
    IndexStorage::Pointer stg =
        InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!stg) {
        cerr << "Failed to create storage " << FLAGS_storage_class
                  << endl;
        return false;
    } else {
        cout << "Created storage " << FLAGS_storage_class << endl;
    }

    // Load Index NOW
    int ret = service->LoadIndex(FLAGS_flat_index, stg);
    if (ret < 0) {
        cerr << "Failed to loadIndex, ret=" << ret << endl;
        return false;
    }
    cout << "Load Index done!" << endl;
    return true;
}

bool LoadIvfIndex(VectorService::Pointer& service) {
    IndexStorage::Pointer stg =
        InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!stg) {
        cerr << "Failed to create storage " << FLAGS_storage_class
                  << endl;
        return false;
    } else {
        cout << "Created storage " << FLAGS_storage_class << endl;
    }

    // Load Index NOW
    int ret = service->LoadIndex(FLAGS_ivf_index, stg);
    if (ret < 0) {
        cerr << "Failed to loadIndex, ret=" << ret << endl;
        return false;
    }
    cout << "Load Index done!" << endl;
    return true;
}

struct RResult{
    size_t flatT = 0;
    size_t ivfT = 0;
    size_t recall = 0;
    size_t pos = 0;
    float acc = 0.0;
};

#include <iomanip>
#include <ctime>
int main(int argc, char *argv[])
{
    std::cout << "gcc: " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << std::endl;

    //gflags
    gflags::SetUsageMessage("Usage: recall [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load plugins first
    LoadPlugin(argc, argv);

    // Create a vector service
    VectorService::Pointer serviceFlat, serviceIvf;
    CreateService(serviceFlat, serviceIvf);
 
    // Init the service
    InitService(serviceFlat);
    InitService(serviceIvf);

    // Load Index using Storage
    LoadFlatIndex(serviceFlat);
    LoadIvfIndex(serviceIvf);

    std::vector<std::pair<std::vector<cat_t>, std::vector<float>>> query;
    LoadQuery(query);
    std::cout << "Query is loaded." << std::endl;

    std::cout << "start search..." << std::endl;
    size_t topK = 200;

    auto eStart = std::chrono::steady_clock::now();

    const size_t threadNum = FLAGS_threads;
    std::vector<std::thread> threadVec;
    std::atomic<size_t> position{0};
    std::vector<RResult> diffVec(query.size());
    const size_t total = query.size();
    for (size_t i = 0; i < threadNum; ++i) {
        threadVec.push_back(std::thread([&serviceFlat, &serviceIvf, &topK, &query, &diffVec, &position, &total]() {
            while(true) {
                auto pos = position.fetch_add(1, std::memory_order_relaxed);
                if (pos >= total) break;
                const size_t anchor = 100000;
                if (pos != 0 && pos % anchor == 0) {
                    std::cout << "query processed: " << pos << std::endl;
                }


    IndexParams params;
    //in order to enable _updateCoarseParam in general_search_context
    params.set(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO, "0.05, 0.05");

    VectorService::SearchContext::Pointer 
            knnContextFlat = serviceFlat->CreateContext(params);
    if (!knnContextFlat) {
            cerr << "Failed to create flat search context" << endl;
            continue;
    }

    VectorService::SearchContext::Pointer 
            knnContextIvf = serviceIvf->CreateContext(params);
    if (!knnContextIvf) {
            cerr << "Failed to create ivf search context" << endl;
            continue;
    }

        auto& p  = query[pos];
        auto catVec(p.first);
        if (catVec.size() == 0) continue;
        catVec.erase(std::unique(catVec.begin(), catVec.end()), catVec.end());
        if (catVec.size() > 20) catVec.resize(20);
        size_t top_k = topK / catVec.size() + 1;

        std::vector<SearchResult> res;
        std::vector<SearchResult> resGT;
        size_t dFlat = 0, dIvf = 0;
        size_t visitedDocLimit = SEARCH_LIMIT;
        auto start = std::chrono::steady_clock::now();

        for (const auto& e: catVec) {
            if (dynamic_cast<GeneralSearchContext*>(knnContextFlat.get())->getVisitedDocNum() >= visitedDocLimit) {
                visitedDocLimit = 0;
                break;
            }
            int ret = serviceFlat->CatKnnSearch(e, top_k, p.second.data(), p.second.size()*sizeof(float), knnContextFlat);
            if (ret < 0) {
                cerr << "Failed to CatKnnSearchFlat, ret=" << ret << endl;
                continue;
            }
            res.insert(res.end(), knnContextFlat->Result().begin(), knnContextFlat->Result().end());
            resGT.insert(resGT.end(), knnContextFlat->Result().begin(), knnContextFlat->Result().end());
        }

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        dFlat = diff.count();


        /*************  ivf ************/
        start = std::chrono::steady_clock::now();

        size_t ivfVisitedCat = 0;
        for (const auto& e: catVec) {
            if (visitedDocLimit == 0 || dynamic_cast<GeneralSearchContext*>(knnContextIvf.get())->getVisitedDocNum() >= visitedDocLimit) { 
                break;
            }
            ++ivfVisitedCat;
            int ret = serviceIvf->CatKnnSearch(e, top_k, p.second.data(), p.second.size()*sizeof(float), knnContextIvf);
            if (ret < 0) {
                cerr << "Failed to CatKnnSearchIvf, ret=" << ret << endl;
                continue;
            }
            res.insert(res.end(), knnContextIvf->Result().begin(), knnContextIvf->Result().end());
        }

        diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        dIvf = diff.count();

        /****************** ivf gt *************/
        dynamic_cast<GeneralSearchContext&>(*knnContextIvf).setCoarseProbeRatio(1.0);
        size_t gtIvfVisitedCat = 0;
        for (const auto& e: catVec) {
            if (gtIvfVisitedCat >= ivfVisitedCat) break;
            ++gtIvfVisitedCat;
            int ret = serviceIvf->CatKnnSearch(e, top_k, p.second.data(), p.second.size()*sizeof(float), knnContextIvf);
            if (ret < 0) {
                cerr << "Failed to GT CatKnnSearchIvf, ret=" << ret << endl;
                continue;
            }
            resGT.insert(resGT.end(), knnContextIvf->Result().begin(), knnContextIvf->Result().end());
        }

        /************** calculate recall rate *************/
        float acc = 0.0;
        if (res.size() == 0 || res.size() != resGT.size()) { 
            std::cerr << res.size() << "->" << resGT.size() << "     ";
            for (const auto& e : catVec) std::cerr << e << " ";
            std::cerr << std::endl;
            std::cerr << "     ";
            for (const auto& e : p.second) std::cerr << e << ",";
            std::cerr << std::endl;
        }
        else {
            size_t correct = 0;
            size_t fi = 0, fj = 0;
            for (; fi < res.size() && fj < resGT.size();)
            {
                if (resGT[fj].key == res[fi].key) {
                    ++correct;
                    ++fi;
                }
                ++fj;
            }
            acc = static_cast<float>(correct) / static_cast<float>(res.size());
            /*
            size_t correct = 0;
            size_t fi = 0, fj = 0;
            for (; fi < res.size() && fj < resGT.size();)
            {
                //if (std::fabs(resGT[fj].score - res[fi].score) < 0.000001) {
                if (almost_equal(resGT[fj].score, res[fi].score, 2)) {
                    ++correct;
                    ++fi;
                }
                ++fj;
            }
            acc = static_cast<float>(correct) / static_cast<float>(res.size());
            */
            /*
            std::unordered_map<cat_t, size_t> rr;
            for (const auto& e : res) ++rr[e.key];
            size_t correct = 0;
            for (const auto& e : resGT) {
                if (rr.count(e.key) > 0 && rr[e.key] > 0) {
                    ++correct;
                    --rr[e.key];
                }
            }
            if (resGT.size() != 0) acc = static_cast<float>(correct) / static_cast<float>(resGT.size());
            */
        }
        /*
        std::unordered_map<cat_t, size_t> rr;
        for (const auto& e : res) ++rr[e.key];
        float acc = 0.0;
        size_t correct = 0;
        for (const auto& e : resGT) {
            if (rr.count(e.key) > 0 && rr[e.key] > 0) {
                ++correct;
                --rr[e.key];
            }
        }
        if (resGT.size() != 0) acc = static_cast<float>(correct) / static_cast<float>(resGT.size());
        */

        diffVec[pos].flatT = dFlat;
        diffVec[pos].ivfT = dIvf;
        diffVec[pos].recall = res.size() != 0 && res.size() == resGT.size() ? res.size() : 0;
        diffVec[pos].pos = pos;
        diffVec[pos].acc = acc;
                }}));
    }

    for(auto& e : threadVec) e.join();

    auto eDiff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - eStart);
    std::cout << "Recall done in ms: " << eDiff.count() << std::endl;

    diffVec.erase(std::remove_if(diffVec.begin(), diffVec.end(), [](RResult e){ return e.recall == 0;}), diffVec.end());

    std::cout << "Valid recall/query: " << diffVec.size() << std::endl;

    /************* latency ****************/
    std::sort(diffVec.begin(), diffVec.end(),
                [](RResult lhs_,
                   RResult rhs_){
                        return lhs_.flatT + lhs_.ivfT <
                                rhs_.flatT + rhs_.ivfT;
                });

    size_t latThd = 50;
    auto latIter = std::upper_bound(diffVec.begin(), diffVec.end(), latThd,[](size_t value_, RResult e_){
        return e_.flatT + e_.ivfT > value_; });
    float latRate = (latIter - diffVec.begin()) / static_cast<float>(diffVec.size());
    size_t c95 = diffVec.size() * 0.95;

    size_t tT = 0;
    float aT = 0.0;
    for(const auto& ii : diffVec) {
        tT += ii.flatT + ii.ivfT;
        aT += ii.acc;
    }
    tT /= diffVec.size();
    aT /= diffVec.size();

    std::cout << "Latency --- min:" << diffVec.front().flatT + diffVec.front().ivfT << " | " << "med:" << diffVec[diffVec.size()/2].flatT + diffVec[diffVec.size()/2].ivfT << " | " << "mea:" << tT << " | 95%:" << diffVec[c95].flatT + diffVec[c95].ivfT << " | " << "max:" << diffVec.back().flatT + diffVec.back().ivfT << " -> " << latRate << std::endl;

    std::ofstream sDiffT("/data0/jiazi/ms_data/diffT.vec");
    for(const auto& e : diffVec) {
        sDiffT << e.flatT  + e.ivfT << " -> " << e.flatT << " -> " << e.ivfT << " -> " << e.recall << " -> " << query[e.pos].first.size() << " -> " << e.acc << " >>> ";
        for(const auto& p : query[e.pos].first) {
            sDiffT << p << " ";
        }
        sDiffT << std::endl;
    }

    /************ acc   *****************/
    std::sort(diffVec.begin(), diffVec.end(),
                [](RResult lhs_,
                   RResult rhs_){
                        return lhs_.acc <
                                rhs_.acc;
                });
    float accThd = 0.95;
    auto accIter = std::lower_bound(diffVec.begin(), diffVec.end(), accThd,[](RResult e_, float value_){
        return e_.acc < value_; });
    float accRate = (diffVec.end() - accIter) / static_cast<float>(diffVec.size());

    std::cout << "Accuracy --- min:" << diffVec.front().acc << " | " << "med:" << diffVec[diffVec.size()/2].acc << " | " << "mea:" << aT << " | " << "95%:" << diffVec[diffVec.size() - c95].acc << " | max:" << diffVec.back().acc << " -> " << accRate << std::endl;

    std::ofstream sDiff("/data0/jiazi/ms_data/diffA.vec");
    for(const auto& e : diffVec) {
        sDiff << e.flatT  + e.ivfT << " -> " << e.flatT << " -> " << e.ivfT << " -> " << e.recall << " -> " << query[e.pos].first.size() << " -> " << e.acc << " >>> ";
        for(const auto& p : query[e.pos].first) {
            sDiff << p << " ";
        }
        sDiff << std::endl;
    }

    // Cleanup
    serviceFlat->UnloadIndex();
    serviceFlat->Cleanup();
    serviceIvf->UnloadIndex();
    serviceIvf->Cleanup();

    return 0;
}
