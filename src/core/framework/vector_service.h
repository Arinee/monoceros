/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     vector_service.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Interface of mercury vector service
 */

#ifndef __MERCURY_VECTOR_SERVICE_H__
#define __MERCURY_VECTOR_SERVICE_H__

#include "index_storage.h"
#include "index_params.h"
#include <memory>
#include <string>
#include <vector>
#include <iostream>

MERCURY_NAMESPACE_BEGIN(core);
class SearchResult;

/*! Vector Service
 */
class VectorService
{
public:
    //! Vector Service Pointer
    typedef std::shared_ptr<VectorService> Pointer;

    /*! Search Context
     */
    class SearchContext
    {
    public:
        //! Search Context Pointer
        typedef std::unique_ptr<SearchContext> Pointer;

        //! Destructor
        virtual ~SearchContext(void) {}

        //! Retrieve search result
        virtual const std::vector<SearchResult> &Result(void) const = 0;
    };

    //! Destructor
    virtual ~VectorService(void) {}

    //! Initialize
    virtual int Init(const IndexParams &params) = 0;

    //! Cleanup
    virtual int Cleanup(void) = 0;

    //! Load index from file path or dir
    virtual int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) = 0;

    virtual int WarmUp(SearchContext::Pointer&) { return 0; }

    //! Unload index
    virtual int UnloadIndex(void) = 0;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) = 0;

    //! Create a context
    virtual SearchContext::Pointer CreateContext(const IndexParams &params) = 0;

    //! KNN Search
    virtual int KnnSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) = 0;

    //! KNN Exhaustive Search
    virtual int ExhaustiveSearch(size_t topk, const void *val,
                                 size_t len, SearchContext::Pointer &context) = 0;

    //! KNN Search
    virtual int CatKnnSearch(cat_t cat_, size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) {
        std::cout << cat_ << topk << val << len << context.get() << std::endl;
        return 0;
    }

    //! KNN Exhaustive Search
    virtual int CatExhaustiveSearch(cat_t cat_, size_t topk, const void *val,
                                 size_t len, SearchContext::Pointer &context) {
        std::cout << cat_ << topk << val << len << context.get() << std::endl;
        return 0;
    }

    //! Add a vector into index, 0 indicates success
    virtual int AddVector(uint64_t key, const void * val, size_t len) = 0;

    //! Delete a vector from index
    virtual int DeleteVector(uint64_t key) = 0;

    //! Update a vector in index
    virtual int UpdateVector(uint64_t key, const void * val, size_t len) = 0;

};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_VECTOR_SERVICE_H__

