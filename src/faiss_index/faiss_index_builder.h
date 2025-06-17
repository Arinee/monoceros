/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     faiss_hnsw_builder.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Builder
 */

#ifndef __MERCURY_FAISS_HNSW_BUILDER_H__
#define __MERCURY_FAISS_HNSW_BUILDER_H__

#include <memory>
#include "framework/index_meta.h"
#include "framework/index_params.h"
#include "framework/vector_holder.h"
#include "framework/index_storage.h"
#include "framework/index_builder.h"

namespace faiss {
class Index;
}

namespace mercury
{

/*! Faiss Index Builder
 */
class FaissIndexBuilder : public IndexBuilder
{
public:
    FaissIndexBuilder()
        :_indexMeta(),
        _faissIndex(nullptr),
        _faissIndexName("unknown")
    {}
    //! Index Builder Pointer
    typedef std::shared_ptr<FaissIndexBuilder> Pointer;

    //! Destructor
    ~FaissIndexBuilder() override = default;

    //! Initialize Builder
    int Init(const IndexMeta &meta, const IndexParams &params) override;

    //! Cleanup Builder
    int Cleanup() override;

    //! Train the data
    int Train(const VectorHolder::Pointer &holder) override;

    //! Build the index
    int BuildIndex(const VectorHolder::Pointer &holder) override;


    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

private:
    IndexMeta _indexMeta;
    faiss::Index *_faissIndex;
    std::string _faissIndexName;
};

} // namespace mercury

#endif // __MERCURY_FAISS_HNSW_BUILDER_H__

