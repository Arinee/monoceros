/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     flat_service.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    interface of mercury flat service
 */

#ifndef __MERCURY_FLAT_SERVICE_H__
#define __MERCURY_FLAT_SERVICE_H__

#include "framework/index_params.h"
#include "framework/search_result.h"
#include "index/base_index_provider.h"
#include "framework/index_params.h"
#include "exhaustive_searcher.h"
#include "flat_searcher.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

/*! Flat Service
 */
class FlatService : public VectorService
{
public:
    //! constructor
    FlatService(void)
        :_indexProvider(nullptr)
    {}

    //! Destructor
    virtual ~FlatService(void) override;

    //! Initialize
    virtual int Init(const IndexParams &params) override;

    //! Cleanup
    virtual int Cleanup(void) override;

    //! Load index from file path or dir
    virtual int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) override;

    //! Unload index
    virtual int UnloadIndex(void) override;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    //! Create a context
    virtual SearchContext::Pointer CreateContext(const IndexParams &params) override;

    //! KNN Search
    virtual int KnnSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;

    //! KNN Exhaustive Search
    virtual int ExhaustiveSearch(size_t topk, const void *val,
                                 size_t len, SearchContext::Pointer &context) override;

    //! Add a vector into index, 0 indicates success
    virtual int AddVector(uint64_t key, const void * val, size_t len) override;

    //! Delete a vector from index
    virtual int DeleteVector(uint64_t key) override;

    //! Update a vector in index
    virtual int UpdateVector(uint64_t key, const void * val, size_t len) override;

protected:
    ExhaustiveSearcher _exhaustiveSearcher;
    FlatSearcher _flatSearcher;
    BaseIndexProvider *_indexProvider;
    IndexMeta _indexMeta;
    IndexParams::Pointer _defaultParams;
};



} // namespace mercury

#endif // __MERCURY_FLAT_SERVICE_H__

