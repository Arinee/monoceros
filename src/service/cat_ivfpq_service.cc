#include "cat_ivfpq_service.h"
#include "index/cat_index_ivfpq.h"

using namespace std;

namespace mercury {

int CatIvfpqService::LoadIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
{
    if (unlikely(!stg)) {
        return IndexError_InvalidArgument;
    }

    auto handlerPtr = stg->open(prefix, false);
    if (!handlerPtr) {
        LOG_WARN("try open [%s] as a index directory.", prefix.c_str());
        handlerPtr = stg->open(prefix + "/" + CAT_IVFPQ_INDEX_FILENAME, false);
        if (!handlerPtr) {
            return IndexError_OpenStorageHandler;
        }
    }

    //TODO only support full data
    Index::Pointer segment_index = CatIndexIvfpq::Pointer(new CatIndexIvfpq);
    if (!_indexProvider->load(move(handlerPtr), segment_index)) {
        LOG_ERROR("IndexProvider load error.");
        return IndexError_LoadPackageIndex;
    }

    if (!_indexProvider->get_index_meta()) {
        return IndexError_Runtime;
    }
    _indexMeta = *(_indexProvider->get_index_meta()); 

    if (_knnSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("IvfpqSearcher Load error.");
        return IndexError_Runtime;
    }

    if (_exhaustiveSearcher.Load(_indexProvider) != 0) {
        LOG_ERROR("ExhaustiveSearcher Load error.");
        return IndexError_Runtime;
    }
    return 0;
}

int CatIvfpqService::AddCatIntoResult(SearchContext::Pointer &context) {
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
    auto* index = dynamic_cast<CatIndexIvfpq*>(indexes.front().get());
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

int CatIvfpqService::KnnSearch(size_t topk, const void *val, size_t len,
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

int CatIvfpqService::ExhaustiveSearch(size_t topk, const void * val,
        size_t len, SearchContext::Pointer &context)
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

INSTANCE_FACTORY_REGISTER_SEARCHER(CatIvfpqService);

};
