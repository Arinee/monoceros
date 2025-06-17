#include "cat_ivfflat_service.h"
#include "framework/index_framework.h"
#include "index/cat_index_ivfflat.h"
#include "index/general_search_context.h"
#include <iostream>

using namespace std;

namespace mercury {

int CatIvfflatService::LoadIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
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
    Index::Pointer segment_index = CatIndexIvfflat::Pointer(new CatIndexIvfflat);
    if (!_indexProvider->load(move(handlerPtr), segment_index)) {
        LOG_ERROR("IndexProvider load error.");
        return IndexError_LoadPackageIndex;
    }

    if (!_indexProvider->get_index_meta()) {
        return IndexError_Runtime;
    }
    _indexMeta = *(_indexProvider->get_index_meta()); 

    if (_knnSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("Searcher Load error.");
        return IndexError_Runtime;
    }
    if (_exhaustiveSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("Searcher Load error.");
        return IndexError_Runtime;
    }
    return 0;
}

int CatIvfflatService::CatKnnSearch(cat_t cat_, size_t topk, const void *val, size_t len,
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
    auto* index = dynamic_cast<CatIndexIvfflat*>(indexes[0].get());
    if(index == nullptr) {
        std::cerr << "Invalid index" << std::endl;
        return -1;
    }

    auto keyCatMap = index->GetKeyCatMap();
    auto catSet = index->GetCatSet();
    cat_t catTmp = 0;
    if (!catSet->find(cat_, catTmp)) return 0;

    CustomFilter filter;
    filter.set([keyCatMap, cat_](key_t key_) {
                    cat_t cat = INVALID_CAT_ID;
                    if (!keyCatMap->findQ(key_, cat)) return true;
                    else return cat != cat_;
                });
    generalContext.setFilter(std::move(filter));

    int ret = _knnSearcher.Search(val, len, topk, &generalContext);
    if (ret != 0) {
        LOG_WARN("Failed to execute search.");
        return IndexError_Runtime;
    }

    return 0;
}

int CatIvfflatService::AddCatIntoResult(SearchContext::Pointer &context) {
    //get index pointer
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
    auto* index = dynamic_cast<CatIndexIvfflat*>(indexes.front().get());
    if(index == nullptr) {
        std::cerr << "Invalid index" << std::endl;
        return -1;
    }

    auto keyCatMap = index->GetKeyCatMap();

    //add cat id into results
    GeneralSearchContext &generalContext = dynamic_cast<GeneralSearchContext &>(*context);
    auto& res = generalContext.Result();
    for (auto& e : res) {
        cat_t cat = INVALID_CAT_ID;
        if (!keyCatMap->findQ(e.key, cat)) continue;
        e.cat = cat;
    }

    return 0;
}

int CatIvfflatService::KnnSearch(size_t topk, const void *val, size_t len,
        SearchContext::Pointer &context)
{
    if (Base::KnnSearch(topk, val, len, context) != 0 ) {
        LOG_ERROR("Failed to call Base::KnnSearch.");
        return -1;
    }

    if (AddCatIntoResult(context) != 0 ) {
        LOG_ERROR("Failed to add cat into result.");
        return -1;
    }

    return 0;
}

int CatIvfflatService::ExhaustiveSearch(size_t topk, const void *val, size_t len,
        SearchContext::Pointer &context)
{
    if (Base::ExhaustiveSearch(topk, val, len, context) != 0 ) {
        LOG_ERROR("Failed to call Base::ExhaustiveSearch.");
        return -1;
    }

    if (AddCatIntoResult(context) != 0 ) {
        LOG_ERROR("Failed to add cat into result.");
        return -1;
    }

    return 0;
}

INSTANCE_FACTORY_REGISTER_SEARCHER(CatIvfflatService);

};


