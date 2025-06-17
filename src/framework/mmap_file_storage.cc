/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mmap_file_storage.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Memory Map File Storage
 */

#include "instance_factory.h"
#include "utility/file.h"
#include "utility/mmap_file.h"

namespace mercury {

/*! Memory Mapping File Storage
 */
struct MMapFileStorage : public IndexStorage
{
    /*! File Handler
     */
    class Handler : public IndexStorage::Handler
    {
    public:
        //! Constructor
        Handler(void) : _mmap_file(), _offset(0) {}

        //! Write data into the storage
        virtual size_t write(const void *data, size_t len);

        //! Write data into the storage
        virtual size_t write(size_t offset, const void *data, size_t len);

        //! Read data from the storage (Zero-copy)
        virtual size_t read(const void **data, size_t len);

        //! Read data from the storage (Zero-copy)
        virtual size_t read(size_t offset, const void **data, size_t len);

        //! Read data from the storage
        virtual size_t read(void *data, size_t len);

        //! Read data from the storage with offset
        virtual size_t read(size_t offset, void *data, size_t len);

        //! Close the writer
        virtual void close(void)
        {
            _mmap_file.close();
        }

        //! Reset the hanlder
        virtual void reset(void)
        {
            _offset = 0;
        }

        //! Retrieve size of file
        virtual size_t size(void) const
        {
            return _mmap_file.region_size();
        }

        //! Create a file
        bool create(const std::string &path, size_t file_size)
        {
            size_t last_slash = path.rfind('/');
            if (last_slash != std::string::npos) {
                File::MakePath(path.substr(0, last_slash).c_str());
            }
            return _mmap_file.create(path.c_str(), file_size);
        }

        //! Open a file
        bool open(const std::string &path, bool rdonly)
        {
            return _mmap_file.open(path.c_str(), rdonly);
        }

    private:
        MMapFile _mmap_file;
        size_t _offset;
    };

    //! Initialize Storage
    virtual int init(const IndexParams &)
    {
        return 0;
    }

    //! Cleanup Storage
    virtual int cleanup(void)
    {
        return 0;
    }

    //! Retrieve non-zero if the storage support random reads
    virtual bool hasRandRead(void) const
    {
        return true;
    }

    //! Retrieve non-zero if the storage support random writes
    virtual bool hasRandWrite(void) const
    {
        return true;
    }

    //! Create a file
    virtual IndexStorage::Handler::Pointer create(const std::string &path,
                                                  size_t size);

    //! Open a file
    virtual IndexStorage::Handler::Pointer open(const std::string &path,
                                                bool rdonly);
};

size_t MMapFileStorage::Handler::write(const void *data, size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (_offset + len >= region_size) {
        len = region_size - _offset;
    }
    std::memcpy((uint8_t *)_mmap_file.region() + _offset, data, len);
    _offset += len;
    return len;
}

size_t MMapFileStorage::Handler::write(size_t offset, const void *data,
                                       size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (offset + len >= region_size) {
        if (offset > region_size) {
            offset = region_size;
        }
        len = region_size - offset;
    }
    std::memcpy((uint8_t *)_mmap_file.region() + offset, data, len);
    return len;
}

size_t MMapFileStorage::Handler::read(const void **data, size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (_offset + len >= region_size) {
        len = region_size - _offset;
    }
    *data = (uint8_t *)_mmap_file.region() + _offset;
    _offset += len;
    return len;
}

size_t MMapFileStorage::Handler::read(size_t offset, const void **data,
                                      size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (offset + len >= region_size) {
        if (offset > region_size) {
            offset = region_size;
        }
        len = region_size - offset;
    }
    *data = (uint8_t *)_mmap_file.region() + offset;
    return len;
}

size_t MMapFileStorage::Handler::read(void *data, size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (_offset + len >= region_size) {
        len = region_size - _offset;
    }
    memcpy(data, (uint8_t *)_mmap_file.region() + _offset, len);
    _offset += len;
    return len;
}

size_t MMapFileStorage::Handler::read(size_t offset, void *data, size_t len)
{
    size_t region_size = _mmap_file.region_size();
    if (offset + len >= region_size) {
        if (offset > region_size) {
            offset = region_size;
        }
        len = region_size - offset;
    }
    memcpy(data, (uint8_t *)_mmap_file.region() + offset, len);
    return len;
}

IndexStorage::Handler::Pointer MMapFileStorage::create(const std::string &path,
                                                       size_t size)
{
    MMapFileStorage::Handler *handler = new (std::nothrow) Handler();
    if (handler) {
        if (!handler->create(path, size)) {
            delete handler;
            handler = nullptr;
        }
    }
    return IndexStorage::Handler::Pointer(handler);
}

IndexStorage::Handler::Pointer MMapFileStorage::open(const std::string &path,
                                                     bool rdonly)
{
    MMapFileStorage::Handler *handler = new (std::nothrow) Handler();
    if (handler) {
        if (!handler->open(path, rdonly)) {
            delete handler;
            handler = nullptr;
        }
    }
    return IndexStorage::Handler::Pointer(handler);
}

INSTANCE_FACTORY_REGISTER_STORAGE(MMapFileStorage);

} // namespace mercury
