/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     cat_flat_service.h
 *   \author   jiazi
 *   \date     Apr 2019
 *   \version  1.0.0
 *   \brief    interface of mercury cat flat service
 */

#ifndef __MERCURY_CAT_FLAT_SERVICE_H__
#define __MERCURY_CAT_FLAT_SERVICE_H__

#include "flat_service.h"

namespace mercury 
{

class CatFlatService : public FlatService
{
public:
    //! Load index from file path or dir
    int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) override;
    int CatKnnSearch(cat_t cat_, size_t topk, const void *val, size_t len,
                    SearchContext::Pointer &context) override;
};



} // namespace mercury

#endif // __MERCURY_CAT_FLAT_SERVICE_H__

