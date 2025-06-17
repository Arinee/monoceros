/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     flat_service.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury flat service
 */

#include "flat_service.h"
#include "framework/index_framework.h"
#include "index/index_flat.h"
#include "index/general_search_context.h"

using namespace std;

namespace mercury {

FlatService::~FlatService() 
{
    Cleanup();
}

int FlatService::Init(const IndexParams &params)
{
    if (_flatSearcher.Init(params) != 0) {
        return IndexError_InvalidArgument;
    }

    if (_exhaustiveSearcher.Init(params) != 0) {
        return IndexError_InvalidArgument;
    }

    _indexProvider = new (std::nothrow) BaseIndexProvider();
    if (!_indexProvider) {
        return IndexError_NoMemory;
    }

    //TODO set method
    IndexDistance::Methods searchMethod = IndexDistance::Methods::kMethodUnknown;
    params.get(PARAM_GENERAL_SEARCHER_SEARCH_METHOD, &searchMethod);

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

int FlatService::Cleanup(void) 
{
    _flatSearcher.Cleanup();
    _exhaustiveSearcher.Cleanup();
    DELETE_AND_SET_NULL(_indexProvider);
    return 0;
}


int FlatService::LoadIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
{
    if (unlikely(!stg)) {
        return IndexError_InvalidArgument;
    }

    auto handlerPtr = stg->open(prefix, false);
    if (!handlerPtr) {
        LOG_WARN("try open [%s] as a index directory.", prefix.c_str());
        handlerPtr = stg->open(prefix + "/" + FLAT_INDEX_FILENAME, false);
        if (!handlerPtr) {
            return IndexError_OpenStorageHandler;
        }
    }

    //TODO only support full data
    Index::Pointer segment_index = IndexFlat::Pointer(new IndexFlat);
    if (!_indexProvider->load(move(handlerPtr), segment_index)) {
        LOG_ERROR("IndexProvider load error.");
        return IndexError_LoadPackageIndex;
    }

    if (!_indexProvider->get_index_meta()) {
        return IndexError_Runtime;
    }
    _indexMeta = *(_indexProvider->get_index_meta()); 

    if (_exhaustiveSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("Searcher Load error.");
        return IndexError_Runtime;
    }

    if (_flatSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("Searcher Load error.");
        return IndexError_Runtime;
    }
    return 0;
}

int FlatService::UnloadIndex(void)
{
    _flatSearcher.Unload();
    _exhaustiveSearcher.Unload();
    if (_indexProvider && _indexProvider->unload() != true) {
        return IndexError_Runtime;
    }
    return 0;
}

int FlatService::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg)
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


VectorService::SearchContext::Pointer FlatService::CreateContext(const IndexParams &params) 
{
    return SearchContext::Pointer(new GeneralSearchContext(params));
}


int FlatService::KnnSearch(size_t topk, const void *val, size_t len,
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
    if (len != _indexMeta.sizeofElement()) {
        LOG_WARN("query size(%lu) error, size of element in index meta is: %lu\n", 
                len, _indexMeta.sizeofElement());
        return IndexError_InvalidArgument;
    }
    GeneralSearchContext &generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    generalContext.clean();

    int ret = _flatSearcher.Search(val, len, topk, &generalContext);
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }
    return 0;
}

int FlatService::ExhaustiveSearch(size_t topk, const void * val,
        size_t len, SearchContext::Pointer &context)
{
    if (topk == 0) {
        LOG_WARN("topk is 0");
        return IndexError_InvalidArgument;
    }
    if (!val) {
        LOG_WARN("query is nullptr");
        return IndexError_InvalidArgument;
    }
    if (len != _indexMeta.sizeofElement()) {
        LOG_WARN("query size(%lu) error, size of element in index meta is: %lu\n", 
                len, _indexMeta.sizeofElement());
        return IndexError_InvalidArgument;
    }
    if (unlikely(!context)) {
        LOG_WARN("context is nullptr");
        return IndexError_InvalidArgument;
    }
    GeneralSearchContext &generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    generalContext.clean();

    int ret = _exhaustiveSearcher.Search(val, len, topk, &generalContext);
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }
    return 0;
}


int FlatService::AddVector(uint64_t key, const void * val, size_t len)
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

int FlatService::DeleteVector(uint64_t key)
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

int FlatService::UpdateVector(uint64_t key, const void * val, size_t len)
{
    (void)key;
    (void)val;
    (void)len;
    // TODO impl
    return IndexError_NotImplemented;
}

INSTANCE_FACTORY_REGISTER_SEARCHER(FlatService);

};


