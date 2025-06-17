/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     ivfpq_index_provider.h
 *   \author   yunheng@xiaohongshu.com
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    IVFPQ PROVIDER
 */

#ifndef __MERCURY_INDEX_IVFPQ_INDEX_PROVIDER_H__
#define __MERCURY_INDEX_IVFPQ_INDEX_PROVIDER_H__

#include "framework/index_framework.h"
#include "index_ivfpq.h"
#include "base_index_provider.h"

namespace mercury {

class IvfpqIndexProvider : public BaseIndexProvider
{
public:
    //get product code
    const uint16_t *getProduct(gloid_t gloid);
};

} // namespace mercury

#endif // __MERCURY_INDEX_IVFPQ_INDEX_PROVIDER_H__
