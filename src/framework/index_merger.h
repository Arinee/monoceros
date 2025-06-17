/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_builder.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Merger
 */

#ifndef __MERCURY_INDEX_REDUCER_H__
#define __MERCURY_INDEX_REDUCER_H__

#include <vector>
#include <string>
#include "framework/index_storage.h"

namespace mercury 
{

class IndexParams;

/*! Index Merger
 */
class IndexMerger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<IndexMerger> Pointer;

    //! Destructor
    virtual ~IndexMerger() {}

    //! Initialize Merger
    virtual int Init(const IndexParams &params) = 0;

    //! Cleanup Merger
    virtual int Cleanup() = 0;

    //! Feed indexes from file paths or dirs
    virtual int FeedIndex(const std::vector<std::string> &prefixes,
                          const IndexStorage::Pointer &stg) = 0;

    //! Merge operator
    virtual int MergeIndex() = 0;

    //! Dump index into file path or dir
    virtual int DumpIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) = 0;
};

} // namespace mercury

#endif // __MERCURY_INDEX_REDUCER_H__

