
#include "gpu_knn_linear.h"
#include "impl/calculator_factory.h"
#include "utils/CopyUtils.cuh"
#include "common/params_define.h"

namespace proxima {
namespace gpu {

static const char kKnnLinearFilename[] = "knn.linear.indexes";
static const char kIndexSegmentMeta[] = "Meta";
static const char kIndexSegmentFeatures[] = "Features";
static const char kIndexSegmentKeys[] = "Keys";

GpuKnnLinearSearcher::GpuKnnLinearSearcher(void)
    : _features(NULL), _feature_keys(NULL), _feature_size(0),
      _features_count(0), _fast_load(false), _file_name(kKnnLinearFilename),
      _device_no(0), _cal(NULL), _index(NULL)
{
}

int GpuKnnLinearSearcher::initImpl(const aitheta::SearcherParams &params)
{
    _fast_load = params.getBool(PARAM_LINEAR_SEARCHER_FASTLOAD);
    std::string file_name = params.getString(PARAM_LINEAR_SEARCHER_FILENAME);
    if (!file_name.empty()) {
        _file_name = std::move(file_name);
    }
    _device_no = params.getInt32(PARAM_LINEAR_SEARCHER_GPU_DEVICE_NO);
    int num_dev = -1;
    CUDA_VERIFY(cudaGetDeviceCount(&num_dev));
    if (_device_no >= num_dev) {
        return -1;
    }
    return 0;
}

int GpuKnnLinearSearcher::cleanupImpl(void)
{
    return 0;
}

int GpuKnnLinearSearcher::loadIndexImpl(
    const std::string &prefix, const aitheta::IndexStorage::Pointer &stg)
{
    if (this->isLoaded()) {
        return aitheta::IndexError_IndexLoaded;
    }

    if (!stg.get()) {
        return aitheta::IndexError_InvalidArgument;
    }

    _handle = stg->open(prefix + '/' + _file_name, true);
    if (!_handle) {
        return aitheta::IndexError_OpenStorageHandler;
    }

    aitheta::IndexPackage package;
    if (!package.load(_handle, !_fast_load)) {
        return aitheta::IndexError_LoadPackageIndex;
    }

    aitheta::IndexPackage::Segment *segment = package.get(kIndexSegmentMeta);
    if (!segment) {
        return aitheta::IndexError_NoMetaFound;
    }

    // Read index meta
    aitheta::IndexMeta index_meta;
    if (!index_meta.deserialize(segment->getData(), segment->getDataSize())) {
        return aitheta::IndexError_LoadIndexMeta;
    }

    // Read features
    segment = package.get(kIndexSegmentFeatures);
    if (!segment) {
        return aitheta::IndexError_NoFeatureFound;
    }
    size_t features_count = segment->getDataSize() / index_meta.sizeofElement();
    if (!features_count) {
        return aitheta::IndexError_InvalidFeatureSize;
    }
    _features = segment->getData();

    // Read features' keys
    segment = package.get(kIndexSegmentKeys);
    if (!segment) {
        return aitheta::IndexError_NoKeyFound;
    }
    size_t keys_count = segment->getDataSize() / sizeof(uint64_t);
    if (!keys_count) {
        return aitheta::IndexError_InvalidKeySize;
    }
    _feature_keys = reinterpret_cast<const uint64_t *>(segment->getData());

    // Mismatch size
    if (keys_count != features_count) {
        return aitheta::IndexError_Mismatch;
    }

    _feature_size = index_meta.sizeofElement();
    _features_count = features_count;

    CalculatorFactory calFactory(index_meta);
    _cal = calFactory.createCal();
    if (_cal == NULL) {
        return aitheta::IndexError_Unsupported;
    }
    _index =
        new LinearIndex(index_meta.dimension(), _feature_size, _cal, &_res);
    if (_index->init(_device_no) < 0) {
        return aitheta::IndexError_InvalidArgument;
    }
    if (_index->add((const char *)_features, _features_count) < 0) {
        return aitheta::IndexError_NoMemory;
    }
    return 0;
}

int GpuKnnLinearSearcher::unloadIndexImpl(void)
{
    _handle.release();
    _features = nullptr;
    _feature_keys = nullptr;
    _feature_size = 0;
    _features_count = 0;
    release();
    return 0;
}

int GpuKnnLinearSearcher::knnSearchImpl(size_t topk, const void *val,
                                        size_t len, Context::Pointer &context)
{
    if (!this->isLoaded()) {
        return aitheta::IndexError_NoIndexLoaded;
    }

    if (!topk || !val || len != _feature_size) {
        return aitheta::IndexError_InvalidArgument;
    }

    GpuKnnLinearSearcherContext &linear_context =
        static_cast<GpuKnnLinearSearcherContext &>(*context);

    // Output final documents
    std::vector<aitheta::IndexSearcher::Document> &result =
        linear_context.result();

    this->knnSearchInLocal(topk, val, 1, linear_context.filter(), &result);
    return 0;
}

void GpuKnnLinearSearcher::knnSearchInLocal(
    size_t topk, const void *query, int qnum,
    const aitheta::IndexSearcher::Filter &filter,
    std::vector<aitheta::IndexSearcher::Document> *result)
{
    result->clear();
    result->reserve(topk);

    // Check arguments
    if (!topk) {
        return;
    }
    if (topk > _features_count) {
        topk = _features_count;
    }

    DeviceScope scope(_device_no);
    std::vector<int> labels(qnum * topk);
    std::vector<float> distances(qnum * topk);
    auto stream = _res.getDefaultStreamCurrentDevice();
    auto query_view =
        proxima::gpu::toDevice<char, 1>(&_res, _device_no, (char *)query,
                                        stream, { int(qnum * _feature_size) });
    auto outDistances = proxima::gpu::toDevice<char, 1>(
        &_res, _device_no, (char *)distances.data(), stream,
        { int(qnum * topk * sizeof(float)) });
    auto outIndices = proxima::gpu::toDevice<int, 1>(
        &_res, _device_no, labels.data(), stream, { int(qnum * topk) });
    int ret = _index->search((const void *)query_view.data(),
                             (float *)outDistances.data(), outIndices.data(),
                             qnum, topk);
    if (ret < 0) {
        return;
    }
    proxima::gpu::fromDevice<float>((float *)outDistances.data(),
                                    (float *)distances.data(), qnum * topk,
                                    stream);
    proxima::gpu::fromDevice<int, 1>(outIndices, (int *)labels.data(), stream);

    for (int i = 0; i < qnum; ++i) {
        int idx = i * topk;
        for (int j = 0; j < topk; j++) {
            uint64_t docid = labels[idx + j];
            uint64_t key = _feature_keys[docid];
            float score = distances[idx + j];
            // if (filter.isValid() && filter(key, score)) {
            //     continue;
            // }
            result->push_back(
                aitheta::IndexSearcher::Document(key, docid, score));
        }
    }
}

void GpuKnnLinearSearcher::release()
{
    if (_cal != NULL) {
        delete _cal;
        _cal = NULL;
    }
    if (_index != NULL) {
        delete _index;
        _index = NULL;
    }
}

INDEX_FACTORY_REGISTER_SEARCHER(GpuKnnLinearSearcher);

} // namespace gpu
} // namespace proxima
