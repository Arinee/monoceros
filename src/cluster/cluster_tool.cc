/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cluster_tool.cc
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Implementation of cluster tool
 */

#include "framework/index_framework.h"
#include "framework/utility/bitmap.h"
#include "framework/utility/file.h"
#include "framework/utility/mmap_file.h"
#include "framework/utility/time_helper.h"
#include "gflags/gflags.h"
#include "kmedoids_cluster.h"
#include "reservoir_sample.h"
#include "multistage_cluster.h"
#include "common/txt_file_holder.h"
#include "utils/index_meta_helper.h"
#include "index/centroid_resource.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

DEFINE_string(input_first_sep, " ", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_int32(meta_dimension, 512, "The dimension of features");
DEFINE_int32(meta_method, 32, "The distance method of features");
DEFINE_int32(classify_method, 32,
             "The distance method of classifying features");
DEFINE_int32(search_method, 32, "The distance method of searching features");
DEFINE_uint32(cluster_count, 2048, "The count of cluster level 1");
DEFINE_uint32(cluster2_count, 100, "The count of cluster level 2");
DEFINE_bool(cluster_only_means, false, "Only means of cluster level 1");
DEFINE_bool(cluster2_only_means, false, "Only means of cluster level 2");
DEFINE_int32(cluster_mean_method, 0, "Mean method of cluster level 1");
DEFINE_int32(cluster2_mean_method, 0, "Mean method of cluster level 2");
DEFINE_double(cluster_bench_ratio, 1.0, "The bench ratio of cluster level 1");
DEFINE_double(cluster2_bench_ratio, 1.0, "The bench ratio of cluster level 2");
DEFINE_bool(enable_suggest, false, "The suggest option of cluster");
DEFINE_uint32(max_iterations, 20, "The max of iterations");
DEFINE_uint32(sample_layers, 5, "The layers of sample");
DEFINE_uint32(query_sample_count, 1200u, "The sample count of queries");
DEFINE_uint32(feature_sample_count, 6000000u, "The sample count of features");
DEFINE_double(search_factor, 0.001, "The factor of search");
DEFINE_uint32(nprobe, 245, "ivf nprobe");

/*! File Stream Holder
 */
class FileStreamHolder : public mercury::VectorHolder
{
public:
    typedef std::shared_ptr<FileStreamHolder> Pointer;

    /*! File Stream Holder Iterator
     */
    class Iterator : public mercury::VectorHolder::Iterator
    {
    public:
        //! Constructor
        Iterator(const FileStreamHolder *holder)
            : _holder(holder), _valid(false), _line_index(0u), _line(),
              _features()
        {
        }

        //! Retrieve pointer of data
        virtual const void *data(void) const
        {
            return _features.data();
        }

        //! Test if the iterator is valid
        virtual bool isValid(void) const
        {
            return _valid;
        }

        //! Retrieve primary key
        virtual uint64_t key(void) const
        {
            return _line_index;
        }

        //! Next iterator
        virtual void next(void)
        {
            this->getline();
        }

        //! Reset the iterator
        virtual void reset(void)
        {
            _line_index = 0u;
            _holder->_stream.seekg(0);
            _line.clear();
        }

        //! Read a line from an input stream
        bool getline(void)
        {
            _valid = !!std::getline(_holder->_stream, _line);
            while (_valid) {
                ++_line_index;
                if (this->parseline()) {
                    break;
                }
                _valid = !!std::getline(_holder->_stream, _line);
            }
            return _valid;
        }

    protected:
        // bool parseline(void)
        // {
        //     size_t left_bracket = _line.find('{');
        //     if (left_bracket == std::string::npos) {
        //         return false;
        //     }
        //     size_t right_bracket = _line.find('}', left_bracket + 1);
        //     if (right_bracket == std::string::npos) {
        //         return false;
        //     }
        //     _features.clear();

        //     std::stringstream ss(_line.substr(
        //         left_bracket + 1, right_bracket - left_bracket - 1));
        //     std::string token;
        //     while (std::getline(ss, token, ',')) {
        //         _features.push_back(std::stof(token));
        //     }
        //     return (_features.size() == _holder->dimension());
        // }

        // bool parseline(void)
        // {
        //     size_t left_bracket = _line.find(';');
        //     if (left_bracket == std::string::npos) {
        //         return false;
        //     }
        //     size_t right_bracket = _line.find(';', left_bracket + 1);
        //     if (right_bracket == std::string::npos) {
        //         return false;
        //     }
        //     _features.clear();

        //     std::stringstream ss(_line.substr(
        //         left_bracket + 1, right_bracket - left_bracket - 1));
        //     std::string token;
        //     while (std::getline(ss, token, ' ')) {
        //         _features.push_back(std::stof(token));
        //     }
        //     return (_features.size() == _holder->dimension());
        // }

        bool parseline(void)
        {
            size_t left_bracket = _line.find(',');
            if (left_bracket == std::string::npos) {
                return false;
            }
            left_bracket = _line.find(',', left_bracket + 1);
            if (left_bracket == std::string::npos) {
                return false;
            }
            left_bracket = _line.find(',', left_bracket + 1);
            if (left_bracket == std::string::npos) {
                return false;
            }

            size_t right_bracket = _line.rfind(',', left_bracket + 1);
            if (right_bracket == std::string::npos) {
                return false;
            }
            _features.clear();

            std::stringstream ss(_line.substr(
                left_bracket + 1, right_bracket - left_bracket - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                _features.push_back(std::stof(token));
            }
            return (_features.size() == _holder->dimension());
        }

    private:
        //! Members
        const FileStreamHolder *_holder;
        bool _valid;
        size_t _line_index;
        std::string _line;
        std::vector<float> _features;

        //! Disable them
        Iterator(void) = delete;
        Iterator(const Iterator &) = delete;
        Iterator(Iterator &&) = delete;
        Iterator &operator=(const Iterator &) = delete;
    };

    //! Destructor
    ~FileStreamHolder(void)
    {
        _stream.close();
    }

    //! Retrieve count of elements in holder
    virtual size_t count(void) const
    {
        return (size_t)-1;
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const
    {
        return FLAGS_meta_dimension;
    }

    //! Retrieve type information
    virtual FeatureType type(void) const
    {
        return mercury::IndexMeta::kTypeFloat;
    }

    //! Create a new iterator
    virtual mercury::VectorHolder::Iterator::Pointer createIterator(void) const
    {
        FileStreamHolder::Iterator *iter = new FileStreamHolder::Iterator(this);
        iter->getline();
        return mercury::VectorHolder::Iterator::Pointer(iter);
    }

    //! Close a file stream
    void close(void)
    {
        _stream.close();
    }

    //! Open a file stream
    bool open(const char *filepath)
    {
        _stream.open(filepath);
        return _stream.is_open();
    }

private:
    mutable std::ifstream _stream;
};

/*! MMap File Holder
 */
class MMapFileHolder : public mercury::VectorHolder
{
public:
    typedef std::shared_ptr<MMapFileHolder> Pointer;

    /*! MMap File Holder Iterator
     */
    class Iterator : public mercury::VectorHolder::Iterator
    {
    public:
        //! Constructor
        Iterator(const void *features, size_t feature_size,
                 size_t features_count)
            : _features(features), _feature_size(feature_size),
              _features_count(features_count), _index(0u)
        {
        }

        //! Retrieve pointer of data
        virtual const void *data(void) const
        {
            return (uint8_t *)_features + _feature_size * _index;
        }

        //! Test if the iterator is valid
        virtual bool isValid(void) const
        {
            return (_index < _features_count);
        }

        //! Retrieve primary key
        virtual uint64_t key(void) const
        {
            return _index;
        }

        //! Next iterator
        virtual void next(void)
        {
            ++_index;
        }

        //! Reset the iterator
        virtual void reset(void)
        {
            _index = 0u;
        }

    private:
        //! Members
        const void *_features;
        size_t _feature_size;
        size_t _features_count;
        size_t _index;

        //! Disable them
        Iterator(void) = delete;
        Iterator(const Iterator &) = delete;
        Iterator(Iterator &&) = delete;
        Iterator &operator=(const Iterator &) = delete;
    };

    //! Destructor
    ~MMapFileHolder(void) {}

    //! Retrieve count of elements in holder
    virtual size_t count(void) const
    {
        return (_file.region_size() % this->sizeofElement());
    }

    //! Retrieve dimension
    virtual size_t dimension(void) const
    {
        return FLAGS_meta_dimension;
    }

    //! Retrieve type information
    virtual FeatureType type(void) const
    {
        return mercury::IndexMeta::kTypeFloat;
    }

    //! Create a new iterator
    virtual mercury::VectorHolder::Iterator::Pointer createIterator(void) const
    {
        return mercury::VectorHolder::Iterator::Pointer(
            new MMapFileHolder::Iterator(_file.region(), this->sizeofElement(),
                                         _file.region_size() /
                                             this->sizeofElement()));
    }

    //! Close a file stream
    void close(void)
    {
        _file.close();
    }

    //! Open a file stream
    bool open(const char *filepath)
    {
        _file.open(filepath, true);
        if (_file.region_size() % this->sizeofElement() != 0) {
            _file.close();
        }
        return _file.isValid();
    }

private:
    mercury::MMapFile _file;
};

/*! Select Document
 */
struct SelectDocument
{
    uint32_t index;      //! Local Index
    float score;         //! Distance Score
    const void *feature; //! Feature pointer

    //! Constructor
    SelectDocument(void) : index(0), score(0.0f), feature(nullptr) {}

    //! Constructor
    SelectDocument(uint32_t i, float v, const void *f)
        : index(i), score(v), feature(f)
    {
    }

    //! Constructor
    SelectDocument(const SelectDocument &rhs)
        : index(rhs.index), score(rhs.score), feature(rhs.feature)
    {
    }

    //! Assignment
    SelectDocument &operator=(const SelectDocument &rhs)
    {
        index = rhs.index;
        score = rhs.score;
        feature = rhs.feature;
        return *this;
    }

    //! Less than
    bool operator<(const SelectDocument &rhs) const
    {
        return (this->score < rhs.score);
    }
};

static inline void UpdateLocalResult(size_t max_size, uint32_t index,
                                     float score, const void *feat,
                                     std::vector<SelectDocument> *heap)
{
    if (heap->size() < max_size) {
        heap->emplace_back(index, score, feat);
        std::push_heap(heap->begin(), heap->end());
    } else {
        if (score < heap->front().score) {
            std::pop_heap(heap->begin(), heap->end());
            heap->pop_back();
            heap->emplace_back(index, score, feat);
            std::push_heap(heap->begin(), heap->end());
        }
    }
}

#if 0
int main(int argc, char *argv[])
{
    gflags::SetUsageMessage("Usage: cluster_tool [options] <input file path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 2) {
        std::cerr << "Usage: cluster_tool [options] <input file path>"
                  << std::endl;
        return 1;
    }

    FileStreamHolder file_holder;
    if (!file_holder.open(argv[1])) {
        std::cerr << "Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }

    mercury::File file;
    file.create("output.indexes", 0);

    for (auto iter = file_holder.createIterator(); iter->isValid();
         iter->next()) {
        file.write(iter->data(), file_holder.sizeofElement());
    }
    return 0;
}
#endif

struct RecallTester
{
    mercury::MultistageCluster *_cluster;
    mercury::KmedoidsCluster *_cluster1; 
    mercury::ClusterFeatureAny *_queries;
    mercury::ClusterFeatureAny *_features;
    mercury::IndexMeta _index_meta;
    size_t _queries_count;
    size_t _features_count;
    size_t _max_selected_count = 50;
    size_t _query_cursor = 0;
    size_t _nprobe = 0;
    double _recall_ratio_sum_top1 = 0.0;
    double _recall_ratio_sum_top10 = 0.0;
    double _recall_ratio_sum_top50 = 0.0;
    double _recall_ratio_sum_top100 = 0.0;
    double _recall_ratio_sum_top200 = 0.0;
    std::mutex _ratio_mutex;

    void updateRecallInThread(mercury::ClusterFeatureAny query)
    {
        std::vector<SelectDocument> query_matrix;
        std::vector<SelectDocument> query_matrix2;
        size_t matched_count = 0;
        // size_t max_matched_count =
        //     (size_t)std::ceil(_features_count * FLAGS_search_factor);

        const auto &centroids_level1 = _cluster->getCentroids();

        for (size_t i = 0; i < centroids_level1.size(); ++i) {
            const auto &item = centroids_level1[i];
            query_matrix.emplace_back(
                i, _index_meta.distance(item.feature(), query), item.feature());
        }
        std::sort(query_matrix.begin(), query_matrix.end());
        
        std::vector<SelectDocument> heap;
        
        for (size_t i_n = 0; i_n < _nprobe; i_n++) {
            const auto &it = query_matrix[i_n];
            const auto &cluster_features = centroids_level1[it.index].similars();

            for (size_t j = 0; j < cluster_features.size(); ++j) {
                float score =
                    _index_meta.distance(cluster_features[j], query);
                UpdateLocalResult(_max_selected_count, j, score,
                                  cluster_features[j], &heap);
                ++matched_count;
                //if (++matched_count == max_matched_count) {
                //    break;
                //}
            }
            
            //if (matched_count == max_matched_count) {
            //    break;
            //}
        }
        
        std::vector<SelectDocument> linear_heap;
        for (size_t i = 0; i < _features_count; ++i) {
            mercury::ClusterFeatureAny feat = _features[i];
            float score = _index_meta.distance(feat, query);
            UpdateLocalResult(_max_selected_count, -1, score, feat,
                              &linear_heap);
        }

        std::sort(heap.begin(), heap.end());
        std::sort(linear_heap.begin(), linear_heap.end());
        
        #if 0
        std::cout << "heap:";
        for(auto &iter: heap){
            std::cout << iter.score << ", ";
        }
        std::cout << std::endl;
        std::cout << "liner heap:";
        for(auto &iter: linear_heap){
            std::cout << iter.score << ", ";
        }
        std::cout << std::endl;
        #endif

        size_t count_top1 = 0, count_top10 = 10, count_top50 = 0, count_top100 = 0,
               count_top200 = 0;
        for (size_t c = 0, k = 0; k < _max_selected_count; ++k) {
            if (std::abs(heap[c].score - linear_heap[k].score) <=
                std::numeric_limits<float>::epsilon()) {
                ++c;
            }
            if (k == 0) {
                count_top1 = c;
            } else if (k == 9) {
                count_top10 = c;
            } else if (k == 49) {
                count_top50 = c;
            } else if (k == 99) {
                count_top100 = c;
            } else if (k == 199) {
                count_top200 = c;
            }
        }

        double ratio_top1 = (double)count_top1 / 1;
        double ratio_top10 = (double)count_top10 / 10;
        double ratio_top50 = (double)count_top50 / 50;
        double ratio_top100 = (double)count_top100 / 100;
        double ratio_top200 = (double)count_top200 / 200;

        std::unique_lock<std::mutex> lock(_ratio_mutex);
        _recall_ratio_sum_top1 += ratio_top1;
        _recall_ratio_sum_top10 += ratio_top10;
        _recall_ratio_sum_top50 += ratio_top50;
        _recall_ratio_sum_top100 += ratio_top100;
        _recall_ratio_sum_top200 += ratio_top200;
        ++_query_cursor;

        LOG_INFO("[%zu][  1] %zu ratio_mean: %f ratio: %f", matched_count,
                 _query_cursor, _recall_ratio_sum_top1 / _query_cursor,
                 ratio_top1);

        LOG_INFO("[%zu][ 10] %zu ratio_mean: %f ratio: %f", matched_count,
                 _query_cursor, _recall_ratio_sum_top10 / _query_cursor,
                 ratio_top10);

        LOG_INFO("[%zu][ 50] %zu ratio_mean: %f ratio: %f", matched_count,
                 _query_cursor, _recall_ratio_sum_top50 / _query_cursor,
                 ratio_top50);

//        LOG_INFO("[%zu][100] %zu ratio_mean: %f ratio: %f", matched_count,
//                 _query_cursor, _recall_ratio_sum_top100 / _query_cursor,
//                 ratio_top100);
//
//        LOG_INFO("[%zu][200] %zu ratio_mean: %f ratio: %f", matched_count,
//                 _query_cursor, _recall_ratio_sum_top200 / _query_cursor,
//                 ratio_top200);
    }

    void run(mercury::ThreadPool &pool)
    {
        for (size_t i = 0; i < _queries_count; ++i) {
            pool.enqueue(
                mercury::Closure::New(this, &RecallTester::updateRecallInThread,
                                      _queries[i]),
                true);
        }
        pool.waitFinish();
    }
};

#if 2
int main(int argc, char *argv[])
{
    gflags::SetUsageMessage("Usage: cluster_tool [options] <input file path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    if (argc < 2) {
        std::cerr << "Usage: cluster_tool [options] <input file path>"
                  << std::endl;
        return 1;
    }

    // Prepare data in holder
    //MMapFileHolder file_holder;
    mercury::TxtFileHolder::Pointer file_holder;
    file_holder.reset(new mercury::TxtFileHolder(mercury::IndexMeta::kTypeFloat, FLAGS_meta_dimension, FLAGS_input_first_sep, FLAGS_input_second_sep));
    if (!file_holder->load(argv[1])) {
        std::cerr << "Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }

    mercury::TxtFileHolder::Pointer query_file_holder;
    query_file_holder.reset(new mercury::TxtFileHolder(mercury::IndexMeta::kTypeFloat, FLAGS_meta_dimension, FLAGS_input_first_sep, FLAGS_input_second_sep));
    std::cout << "argv[2]:" << argv[2] << std::endl;
    if (!query_file_holder->load(argv[2])) {
        std::cerr << "Cannot open test query file: " << argv[2] << std::endl;
        return 1;
    }
    LOG_INFO("FLAGS_meta_dimension:       %d", FLAGS_meta_dimension);
    LOG_INFO("FLAGS_meta_method:          %d", FLAGS_meta_method);
    LOG_INFO("FLAGS_classify_method:      %d", FLAGS_classify_method);
    LOG_INFO("FLAGS_search_method:        %d", FLAGS_search_method);
    LOG_INFO("FLAGS_cluster_count:        %u", FLAGS_cluster_count);
    LOG_INFO("FLAGS_cluster_only_means:   %u", FLAGS_cluster_only_means);
    LOG_INFO("FLAGS_cluster_mean_method:  %u", FLAGS_cluster_mean_method);
    LOG_INFO("FLAGS_cluster_bench_ratio:  %f", FLAGS_cluster_bench_ratio);
    LOG_INFO("FLAGS_cluster2_count:       %u", FLAGS_cluster2_count);
    LOG_INFO("FLAGS_cluster2_only_means:  %u", FLAGS_cluster2_only_means);
    LOG_INFO("FLAGS_cluster2_mean_method: %u", FLAGS_cluster2_mean_method);
    LOG_INFO("FLAGS_cluster2_bench_ratio: %f", FLAGS_cluster2_bench_ratio);
    LOG_INFO("FLAGS_sample_layers:        %u", FLAGS_sample_layers);
    LOG_INFO("FLAGS_enable_suggest:       %u", FLAGS_enable_suggest);
    LOG_INFO("FLAGS_max_iterations:       %u", FLAGS_max_iterations);
    LOG_INFO("FLAGS_query_sample_count:   %u", FLAGS_query_sample_count);
    LOG_INFO("FLAGS_feature_sample_count: %u", FLAGS_feature_sample_count);
    LOG_INFO("FLAGS_search_factor:        %f", FLAGS_search_factor);
    
    mercury::IndexMeta index_meta;
    index_meta.setMeta(file_holder->type(), file_holder->dimension());
    index_meta.setMethod(
        static_cast<mercury::IndexDistance::Methods>(FLAGS_meta_method));
    std::cout << mercury::IndexMetaHelper::toString(index_meta) << std::endl;    

    std::vector<mercury::ClusterFeatureAny> feature_list;
    std::vector<mercury::ClusterFeatureAny> query_list;
    std::vector<mercury::ClusterFeatureAny> raw_feature_list; 

    {
        
        for (auto iter = query_file_holder->createIterator(); iter->isValid();
             iter->next()) {
            size_t roughElemSize = index_meta.sizeofElement();
            char* buf = (char *)malloc(sizeof(char) * roughElemSize);
            memcpy(buf, iter->data(), roughElemSize);
            query_list.push_back(buf);
        }
        
        for (auto iter = file_holder->createIterator(); iter->isValid();
             iter->next()) {
            size_t roughElemSize = index_meta.sizeofElement();
            char* buf = (char *)malloc(sizeof(char) * roughElemSize);
            memcpy(buf, iter->data(), roughElemSize);
            raw_feature_list.push_back(buf);
        }
        LOG_INFO("raw_feature_list: %zu", raw_feature_list.size());
        /*
        std::random_device rd;
        std::mt19937 mt(rd());
        
        for (size_t i = 0; i < FLAGS_query_sample_count; ++i) {
            std::uniform_int_distribution<size_t> query_dist(
                0, raw_feature_list.size() - 1);
            size_t val = query_dist(mt);

            //query_list.push_back(raw_feature_list[val]);
            raw_feature_list[val] = raw_feature_list.back();
            raw_feature_list.resize(raw_feature_list.size() - 1);
        }
        */
        mercury::ReservoirSample(raw_feature_list, FLAGS_feature_sample_count,
                                 &feature_list);

        LOG_INFO("query_list:       %zu", query_list.size());
        LOG_INFO("feature_list:     %zu", feature_list.size());
    }


    mercury::MultistageCluster cluster;
    mercury::KmedoidsCluster cluster1;
    mercury::KmedoidsCluster cluster2;

    cluster1.setClusterCount(FLAGS_cluster_count);
    cluster1.setOnlyMeans(FLAGS_cluster_only_means);
    cluster1.setDistance(index_meta.measure());
    
    //cluster2.setClusterCount(FLAGS_cluster2_count);
    //cluster2.setMaxIterations(FLAGS_max_iterations);
    //cluster2.setOnlyMeans(FLAGS_cluster2_only_means);
    //cluster2.setEnableSuggest(FLAGS_enable_suggest);
    //cluster2.setBenchRatio(FLAGS_cluster2_bench_ratio);
    //cluster2.setDistance(index_meta.measure());
    //cluster2.setMeanMethod(static_cast<mercury::KmedoidsCluster::MeanMethods>(FLAGS_cluster2_mean_method));

    mercury::ThreadPool thread_pool;
    mercury::ElapsedTime stamp;

    // clustering first
    LOG_DEBUG("Start clustering, list %zu", feature_list.size());
    cluster.mount(feature_list.data(), feature_list.size(), index_meta.type(),
                  index_meta.sizeofElement());
    bool res = cluster.cluster(thread_pool, cluster1);
    LOG_DEBUG("res: %d, Cluster Count: %zu, Cost: %f, Elapsed: %zu ms",
              res, cluster.getCentroids().size(),
              cluster.getCentroidsCost(), stamp.update());

    // classifying all features
    index_meta.setMethod(
        static_cast<mercury::IndexDistance::Methods>(FLAGS_classify_method));
    cluster1.setDistance(index_meta.measure());

    LOG_DEBUG("Start classifing, list %zu", raw_feature_list.size());
    cluster.mount(raw_feature_list.data(), raw_feature_list.size(),
                  index_meta.type(), index_meta.sizeofElement());
    res = cluster.classify(thread_pool, cluster1);
    LOG_DEBUG("res:%d, Classify: %zu, Elapsed: %zu ms",
              int(res), raw_feature_list.size(), stamp.update());
    
    size_t roughElemSize = index_meta.sizeofElement();
    mercury::CentroidResource::RoughMeta roughMeta(roughElemSize, 1, {FLAGS_cluster_count});
    mercury::CentroidResource::IntegrateMeta integrateMeta(0,0,0);    

    mercury::CentroidResource::Pointer _resource;
    _resource.reset(new mercury::CentroidResource);
    if (!_resource->create(roughMeta, integrateMeta)) {
        LOG_ERROR("failed to create centroid resource.");
        return -1;
    }

    size_t centroidIndex = 0;
    for (const auto &it : cluster.getCentroids()) {
        _resource->setValueInRoughMatrix(0, centroidIndex, it.feature());
        printf("centroid index:%ld, value:", centroidIndex);
        for(size_t debug_i = 0; debug_i < index_meta.sizeofElement(); debug_i+=4){
            float * debug = (float*)(((char *)it.feature()) + debug_i);
            printf("%f,", *debug);
        }
        printf("\n");
        centroidIndex++;
    }
    _resource->DumpToFile("rough_matrix.recall", "integrate_matrix.recall");

    _resource.reset(new mercury::CentroidResource);
    mercury::MMapFile roughFile, integrateFile;
    roughFile.open("rough_matrix.recall", true);
    integrateFile.open("integrate_matrix", true);

    bool bret = _resource->init((void *)roughFile.region(),
            roughFile.region_size(),
            (void *)integrateFile.region(), 
            integrateFile.region_size());
    if (!bret) {
        LOG_ERROR("centroid resource init error");
        return false;
    }
    
    // print 
    for(size_t i = 0; i < _resource->getLeafCentroidNum(); i++){
        printf("centroid index:%ld, value:", i);
        for(size_t debug_i = 0; debug_i < index_meta.sizeofElement(); debug_i+=4){
            float* debug = (float *)(((char *)_resource->getValueInRoughMatrix(0, i)) + debug_i);
            printf("%f,", *debug);
        }
        printf("\n");
    }
    // Recall tester
    RecallTester tester;
    tester._cluster = &cluster;
    tester._cluster1 = &cluster1;
    tester._queries = query_list.data();
    tester._features = raw_feature_list.data();
    tester._queries_count = query_list.size();
    tester._features_count = raw_feature_list.size();
    tester._index_meta = index_meta;
    tester._nprobe = FLAGS_nprobe;
    tester._index_meta.setMethod(
        static_cast<mercury::IndexDistance::Methods>(FLAGS_search_method));
    tester.run(thread_pool);
    
    // Exit
    for(auto &buf : raw_feature_list){
        free(const_cast<void *>(buf));
    }
    return 0;
}
#endif
