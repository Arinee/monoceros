/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cat_ivfflat_service.h
 *   \author   jiazi
 *   \date     Apr 2019
 *   \version  1.0.0
 *   \brief    interface of mercury cat ivfflat service
 */

#ifndef __MERCURY_CAT_IVFFLAT_SERVICE_H__
#define __MERCURY_CAT_IVFFLAT_SERVICE_H__

#include "ivfflat_service.h"

namespace mercury 
{

class CatIvfflatService : public IvfflatService
{
public:
    //! Load index from file path or dir
    int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) override;

    int CatKnnSearch(cat_t cat_, size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;

    int KnnSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;

    int ExhaustiveSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;
private:
    int AddCatIntoResult(SearchContext::Pointer &context);
    using Base = IvfflatService;
};

} // namespace mercury

#endif // __MERCURY_CAT_IVFFLAT_SERVICE_H__

