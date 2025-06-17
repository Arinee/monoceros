/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cat_flat_service.cc
 *   \author   jiazi
 *   \date     Apr 2019
 *   \version  1.0.0
 *   \brief    interface of mercury cat flat service
 */

#include "cat_flat_service.h"
#include "framework/index_framework.h"
#include "index/cat_index_flat.h"
#include "index/general_search_context.h"
#include <iostream>

using namespace std;

namespace mercury {

int CatFlatService::LoadIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
{
    if (unlikely(!stg)) {
        return IndexError_InvalidArgument;
    }

    auto handlerPtr = stg->open(prefix, true);
    if (!handlerPtr) {
        LOG_WARN("try open [%s] as a index directory.", prefix.c_str());
        handlerPtr = stg->open(prefix + "/output.indexes", false);
        if (!handlerPtr) {
            return IndexError_OpenStorageHandler;
        }
    }

    //TODO only support full data
    Index::Pointer segment_index = CatIndexFlat::Pointer(new CatIndexFlat);
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

int CatFlatService::CatKnnSearch(cat_t cat_, size_t topk, const void *val, size_t len,
        SearchContext::Pointer &context) {
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
        LOG_WARN("query size(%lu) error, size of element in index meta is:%lu\n", len, _indexMeta.sizeofElement());
        return IndexError_InvalidArgument;
    }
    auto& generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    generalContext.clean();

    if (!_indexProvider) {
        std::cerr << "Invalid index provider" << std::endl;
        return -1;
    }
    auto segments = _indexProvider->get_segment_list();
    auto indexes = _indexProvider->get_segment_list();
    if (segments.size() != 1) {
        std::cerr << "Unsupported seg size: " << segments.size() << std::endl;
        return -1;
    }
    auto* index = dynamic_cast<CatIndexFlat *>(indexes[0].get());
    if(index == nullptr) {
        std::cerr << "Invalid index" << std::endl;
        return -1;
    }

    auto postIter = index->GetPostingIter(cat_);
    CustomDocFeeder feeder;
    feeder.set([postIter]() mutable { return postIter.next(); });
    generalContext.setDocFeeder(std::move(feeder));

    int ret = _flatSearcher.Search(val, len, topk, &generalContext);
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }

    return 0;
}

INSTANCE_FACTORY_REGISTER_SEARCHER(CatFlatService);

};


