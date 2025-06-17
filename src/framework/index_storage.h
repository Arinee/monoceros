/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_storage.h
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Storage
 */

#ifndef __MERCURY_INDEX_STORAGE_H__
#define __MERCURY_INDEX_STORAGE_H__

#include "index_params.h"
#include <memory>
#include <string>

namespace mercury 
{

/*! Index Storage
 */
struct IndexStorage
{
    //! Index Storage Pointer
    typedef std::shared_ptr<IndexStorage> Pointer;

    /*! Storage Handler
     */
    struct Handler
    {
        //! Index Storage Handler Pointer
        typedef std::unique_ptr<Handler> Pointer;

        //! Destructor
        virtual ~Handler(void) {}

        //! Close the hanlder and cleanup resource
        virtual void close(void) = 0;

        //! Reset the hanlder (include resource)
        virtual void reset(void) = 0;

        //! Write data into the storage
        virtual size_t write(const void *data, size_t len) = 0;

        //! Write data into the storage with offset
        virtual size_t write(size_t offset, const void *data, size_t len) = 0;

        //! Read data from the storage (Zero-copy)
        virtual size_t read(const void **data, size_t len) = 0;

        //! Read data from the storage with offset (Zero-copy)
        virtual size_t read(size_t offset, const void **data, size_t len) = 0;

        //! Read data from the storage
        virtual size_t read(void *data, size_t len) = 0;

        //! Read data from the storage with offset
        virtual size_t read(size_t offset, void *data, size_t len) = 0;

        //! Retrieve size of file
        virtual size_t size(void) const = 0;
    };

    //! Destructor
    virtual ~IndexStorage(void) {}

    //! Initialize Storage
    virtual int init(const IndexParams &params) = 0;

    //! Cleanup Storage
    virtual int cleanup(void) = 0;

    //! Create a file with size
    virtual Handler::Pointer create(const std::string &path, size_t size) = 0;

    //! Open a file
    virtual Handler::Pointer open(const std::string &path, bool rdonly) = 0;

    //! Retrieve non-zero if the storage support random reads
    virtual bool hasRandRead(void) const = 0;

    //! Retrieve non-zero if the storage support random writes
    virtual bool hasRandWrite(void) const = 0;
};

} // namespace mercury

#endif // __MERCURY_INDEX_STORAGE_H__

