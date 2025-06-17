#include "faiss_index_builder.h"
#include "faiss/AutoTune.h"
#include "faiss/index_io.h"
#include "framework/index_framework.h"
#include "framework/index_logger.h"
#include "storage_io_writer.h"
#include <vector>
#include <string>

using namespace std;

namespace mercury {


//! Initialize Builder
int FaissIndexBuilder::Init(const IndexMeta &meta, const IndexParams &params)
{
    _indexMeta = meta;
    int dim = (int)_indexMeta.dimension();
    //NOTE only support L2 and IP
    faiss::MetricType metricType = faiss::METRIC_L2;
    if (_indexMeta.type() == IndexMeta::kTypeFloat) {
        if (_indexMeta.method() == IndexDistance::kMethodFloatSquaredEuclidean) {
            metricType = faiss::METRIC_L2;
        } else if (_indexMeta.method() == IndexDistance::kMethodFloatInnerProduct) {
            metricType = faiss::METRIC_INNER_PRODUCT;
        }
    } else {
        LOG_ERROR("not supported meta type.");
        return -1;
    }

    string faissIndexName = params.getString(PARAM_FAISS_INDEX_NAME);
    if (faissIndexName.empty()) {
        LOG_ERROR("faiss index name is empty. please set params: %s", PARAM_FAISS_INDEX_NAME.c_str());
        return -1;
    } else if (faissIndexName.find("IDMap") == string::npos) {
        LOG_ERROR("mercury support faiss index with IDMap.");
        return -1;
    } else {
        // let's suppose it is ready;
    }
    _faissIndexName = faissIndexName;

    _faissIndex = faiss::index_factory(dim, _faissIndexName.c_str(), metricType);
    if (!_faissIndex) {
        LOG_ERROR("failed to create faiss index, name: %s", _faissIndexName.c_str());
        return -1;
    }
    return 0;
}

//! Cleanup Builder
int FaissIndexBuilder::Cleanup(void)
{
    if (_faissIndex) {
        delete _faissIndex;
        _faissIndex = nullptr;
    }
    return 0;
}


//TODO faiss中的train的中间结果不可以dump出来?
//! Train the data
int FaissIndexBuilder::Train(const VectorHolder::Pointer &holder)
{
    VectorHolder::Iterator::Pointer iter = holder->createIterator();
    if (!iter) {
        LOG_ERROR("create iterator error.");
        return -1;
    }
    vector<float> vectorData;
    size_t n = 0;
    size_t dim = _indexMeta.dimension();
    for (; iter->isValid(); iter->next()) {
        auto feature = (float *)iter->data();
        for (size_t i = 0; i < dim; ++i) {
            vectorData.push_back(feature[i]);
        }
        ++n;
    }
    LOG_INFO("train on dataset, dim: %lu n: %lu", dim, n);
    _faissIndex->train(n, vectorData.data());
    return 0;
}


//! Build the index
//TODO only float

int FaissIndexBuilder::BuildIndex(const VectorHolder::Pointer& holder)
{
    VectorHolder::Iterator::Pointer iter = holder->createIterator();
    if (!iter) {
        LOG_ERROR("create iterator error.");
        return -1;
    }
    //NOTE for batch mode
    while (iter->isValid()) {
        vector<long> keys;
        vector<float> values;
        size_t cursor = 0;
        for (; iter->isValid() && cursor < 1000; iter->next()) {
            keys.push_back((long)iter->key());
            auto feature = (float *) iter->data();
            for (size_t d = 0; d < _indexMeta.dimension(); ++d) {
                values.push_back(feature[d]);
            }
        }
        if (!keys.empty()) {
            _faissIndex->add_with_ids(keys.size(), values.data(), keys.data());
        }
    }

    return 0;
}


//! Dump index into file or memory
int FaissIndexBuilder::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg)
{
    // NOTE: can not get size of file beforehand, set to 0
    // only support FileStorage
    IndexStorage::Handler::Pointer handler = stg->create(path, 0);
    if (!handler) {
        LOG_ERROR("index storage failed to create handler");
        return -1;
    }
    LOG_INFO("dump index to %s", path.c_str());
    StorageIOWriter ioWriter(handler);
    faiss::write_index(_faissIndex, &ioWriter);
    return 0;
}


INSTANCE_FACTORY_REGISTER_BUILDER(FaissIndexBuilder);

}; // namespace mercury
