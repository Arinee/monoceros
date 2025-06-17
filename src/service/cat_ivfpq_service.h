#pragma once

#include "ivfpq_service.h"

namespace mercury 
{

/*! Ivfpq Service
 */
class CatIvfpqService : public IvfpqService
{
public:
    //! Load index from file path or dir
    int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) override;

    //! KNN Search
    int KnnSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;

    //! KNN Exhaustive Search
    int ExhaustiveSearch(size_t topk, const void *val,
                                 size_t len, SearchContext::Pointer &context) override;

private:
    int AddCatIntoResult(SearchContext::Pointer &context);
    using Base = IvfpqService;
};

} // namespace mercury
