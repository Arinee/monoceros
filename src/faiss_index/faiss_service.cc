#include "faiss_service.h"
#include "framework/index_framework.h"
#include "index_faiss.h"
#include "index/general_search_context.h"

using namespace std;

namespace mercury {

FaissService::~FaissService() 
{
    Cleanup();
}

int FaissService::Init(const IndexParams &params)
{
    _indexProvider = new (std::nothrow) FaissIndexProvider();
    if (!_indexProvider) {
        return IndexError_NoMemory;
    }

    size_t incrSegmentDocNum = params.getUint64(PARAM_GENERAL_SEARCHER_INCR_SEGMENT_DOC_NUM);
    if (incrSegmentDocNum <= 0) {
        incrSegmentDocNum = 10000ul;
    }
    std::string incrSegmentPath = params.getString(PARAM_GENERAL_SEARCHER_INCR_SEGMENT_PATH);
    if (incrSegmentPath.empty()) {
        incrSegmentPath = ".";
    }

    _defaultParams = IndexParams::Pointer(new IndexParams(params));

    if (!_indexProvider->init(incrSegmentDocNum, incrSegmentPath, _defaultParams)) {
        return IndexError_Runtime;
    }
    return 0;
}

int FaissService::Cleanup(void) 
{
    DELETE_AND_SET_NULL(_indexProvider);
    return 0;
}


int FaissService::LoadIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
{
    if (unlikely(!stg)) {
        return IndexError_InvalidArgument;
    }

    auto handlerPtr = stg->open(prefix, false);
    if (!handlerPtr) {
        LOG_WARN("try open [%s] as a index directory.", prefix.c_str());
        handlerPtr = stg->open(prefix + "/output.indexes", false);
        if (!handlerPtr) {
            return IndexError_OpenStorageHandler;
        }
    }

    //TODO only support full data
    Index::Pointer segment_index = IndexFaiss::Pointer(new IndexFaiss);
    if (!_indexProvider->load(move(handlerPtr), segment_index)) {
        LOG_ERROR("IndexProvider load error.");
        return IndexError_LoadPackageIndex;
    }

    /*
    if (!_indexProvider->get_index_meta()) {
        return IndexError_Runtime;
    }
    _indexMeta = *(_indexProvider->get_index_meta()); 
    */

    return 0;
}

int FaissService::UnloadIndex(void)
{
    if (_indexProvider && _indexProvider->unload() != true) {
        return IndexError_Runtime;
    }
    return 0;
}

int FaissService::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg)
{
    // TODO multi segment
    if (_indexProvider->get_segment_list().size() != 1) {
        LOG_WARN("Dont support multi segment dump.");
        return IndexError_NotImplemented;
    }
    Index::Pointer index = _indexProvider->get_segment_list()[0];
    if (!index) {
        LOG_WARN("Index in index provider is nullptr");
        return IndexError_Runtime;
    }
    // only_dump_meta=false
    if (index->Dump(stg, path, false) != false) {
        LOG_WARN("Index dump return error.");
        return IndexError_Runtime;
    }
    return 0;
}


VectorService::SearchContext::Pointer FaissService::CreateContext(const IndexParams &params) 
{
    return SearchContext::Pointer(new GeneralSearchContext(params));
}


int FaissService::KnnSearch(size_t topk, const void *val, size_t len,
        SearchContext::Pointer &context)
{
    if (topk == 0) {
        LOG_WARN("topk is 0");
        return IndexError_InvalidArgument;
    }
    if (!val) {
        LOG_WARN("query is nullptr");
        return IndexError_InvalidArgument;
    }
    if (unlikely(!context)) {
        LOG_WARN("context is nullptr");
        return IndexError_InvalidArgument;
    }
    //ToDo: this should be filled in segment
    (void)len;
    /*
    if (len != _indexMeta.sizeofElement()) {
        LOG_WARN("query size(%lu) error, size of element in index meta is: %lu\n", 
                len, _indexMeta.sizeofElement());
        return IndexError_InvalidArgument;
    }
    */
    GeneralSearchContext &generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    generalContext.clean();

    std::vector<float> dist(topk);
    std::vector<long> label(topk);
    const auto& indexes = _indexProvider->get_segment_list();
    int ret = 0;
    for (const auto& e : indexes) {
        ret = dynamic_cast<IndexFaiss*>(e.get())->Search(val, topk, dist.data(), label.data());
    }
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }

    for (size_t i = 0; i < topk; ++i) {
        generalContext.emplace_back(_indexProvider->getPK(label[i]), label[i], dist[i]);
    }

    return 0;
}

int FaissService::ExhaustiveSearch(size_t topk, const void * val,
        size_t len, SearchContext::Pointer &context)
{
    LOG_WARN("Fake... Not implemented yet...");
    if (topk == 0) {
        LOG_WARN("topk is 0");
        return IndexError_InvalidArgument;
    }
    if (!val) {
        LOG_WARN("query is nullptr");
        return IndexError_InvalidArgument;
    }
    if (unlikely(!context)) {
        LOG_WARN("context is nullptr");
        return IndexError_InvalidArgument;
    }
    //ToDo: this should be filled in segment
    (void)len;
    /*
    if (len != _indexMeta.sizeofElement()) {
        LOG_WARN("query size(%lu) error, size of element in index meta is: %lu\n", 
                len, _indexMeta.sizeofElement());
        return IndexError_InvalidArgument;
    }
    */
    GeneralSearchContext &generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    generalContext.clean();

    std::vector<float> dist(topk);
    std::vector<long> label(topk);
    const auto& indexes = _indexProvider->get_segment_list();
    int ret = 0;
    for (const auto& e : indexes) {
        ret = dynamic_cast<IndexFaiss*>(e.get())->Search(val, topk, dist.data(), label.data());
    }
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }

    for (size_t i = 0; i < topk; ++i) {
        generalContext.emplace_back(_indexProvider->getPK(label[i]), label[i], dist[i]);
    }

    //LOG_WARN("Failed to execute search.");
    //return IndexError_Runtime;

    return 0;
}


int FaissService::AddVector(uint64_t key, const void * val, size_t len)
{
    if (unlikely(!val || len == 0)) {
        LOG_ERROR("Add vector can't be nullptr");
        return IndexError_InvalidArgument;
    }

    if (unlikely(!_indexProvider)) {
        return IndexError_Runtime;
    }
    docid_t docid = _indexProvider->addVector(key, val, len);
    if (docid == INVALID_DOCID) {
        return IndexError_Runtime;
    }
    return 0;
}

int FaissService::DeleteVector(uint64_t key)
{
    if (unlikely(!_indexProvider)) {
        return IndexError_Runtime;
    }
    bool bret = _indexProvider->deleteVector(key);
    if (bret != true) {
        return IndexError_Runtime;
    }
    return 0;
}

int FaissService::UpdateVector(uint64_t key, const void * val, size_t len)
{
    (void)key;
    (void)val;
    (void)len;
    // TODO impl
    return IndexError_NotImplemented;
}

INSTANCE_FACTORY_REGISTER_SEARCHER(FaissService);

};
