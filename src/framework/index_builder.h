/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_builder.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Builder
 */

#ifndef __MERCURY_INDEX_BUILDER_H__
#define __MERCURY_INDEX_BUILDER_H__

#include <memory>
#include "framework/index_storage.h"
#include "framework/vector_holder.h"

namespace mercury 
{

class IndexParams;
class IndexMeta;

/*! Index Builder
 */
class IndexBuilder
{
public:
    //! Index Builder Pointer
    typedef std::shared_ptr<IndexBuilder> Pointer;

    //! Destructor
    virtual ~IndexBuilder() = default;

    //! Initialize Builder
    virtual int Init(const IndexMeta &meta, const IndexParams &params) = 0;

    //! Cleanup Builder
    virtual int Cleanup() = 0;

    //! Train the data
    virtual int Train(const VectorHolder::Pointer &holder) = 0;

    //! Build the index
    virtual int BuildIndex(const VectorHolder::Pointer &holder) = 0;

    //! Dump index into file or memory
    virtual int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) = 0;
};

} // namespace mercury

#endif // __MERCURY_INDEX_BUILDER_H__

