/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     extra_memory_storage.h
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Extra Memory Storage
 */

#ifndef __MERCURY_EXTRA_MEMORY_STORAGE_H__
#define __MERCURY_EXTRA_MEMORY_STORAGE_H__

#include "framework/index_storage.h"
#include <cstring>
#include <map>

namespace mercury {

/*! Extra Memory Storage
 */
struct ExtraMemoryStorage : public IndexStorage
{
    //! Extra Memory Storage Pointer
    typedef std::shared_ptr<ExtraMemoryStorage> Pointer;

    /*! Extra Memory Storage Handler
     */
    class Handler : public IndexStorage::Handler
    {
    public:
        //! Extra Memory Storage Handler Pointer
        typedef std::unique_ptr<Handler> Pointer;

        //! Constructor
        Handler(void *region, size_t len)
            : _region(region), _region_size(len), _offset(0)
        {
        }

        //! Write data into the storage
        virtual size_t write(const void *data, size_t len)
        {
            if (_offset + len >= _region_size) {
                len = _region_size - _offset;
            }
            std::memcpy((uint8_t *)_region + _offset, data, len);
            _offset += len;
            return len;
        }

        //! Write data into the storage
        virtual size_t write(size_t offset, const void *data, size_t len)
        {
            if (offset + len >= _region_size) {
                if (offset > _region_size) {
                    offset = _region_size;
                }
                len = _region_size - offset;
            }
            std::memcpy((uint8_t *)_region + offset, data, len);
            _offset = offset + len;
            return len;
        }

        //! Read data from the storage (Zero-copy)
        virtual size_t read(const void **data, size_t len)
        {
            if (_offset + len >= _region_size) {
                len = _region_size - _offset;
            }
            *data = (uint8_t *)_region + _offset;
            _offset += len;
            return len;
        }

        //! Read data from the storage (Zero-copy)
        virtual size_t read(size_t offset, const void **data, size_t len)
        {
            if (offset + len >= _region_size) {
                if (offset > _region_size) {
                    offset = _region_size;
                }
                len = _region_size - offset;
            }
            *data = (uint8_t *)_region + offset;
            _offset = offset + len;
            return len;
        }

        //! Read data from the storage
        size_t read(void *data, size_t len)
        {
            if (_offset + len >= _region_size) {
                len = _region_size - _offset;
            }
            memcpy(data, (uint8_t *)_region + _offset, len);
            _offset += len;
            return len;
        }

        //! Read data from the storage with offset
        size_t read(size_t offset, void *data, size_t len)
        {
            if (offset + len >= _region_size) {
                if (offset > _region_size) {
                    offset = _region_size;
                }
                len = _region_size - offset;
            }
            memcpy(data, (uint8_t *)_region + offset, len);
            return len;
        }

        //! Close the writer
        virtual void close(void)
        {
            _region = nullptr;
            _region_size = 0;
            _offset = 0;
        }

        //! Reset the hanlder
        virtual void reset(void)
        {
            _offset = 0;
        }

        //! Retrieve size of file
        virtual size_t size(void) const
        {
            return _region_size;
        }

    private:
        void *_region;
        size_t _region_size;
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

    //! Create a file
    virtual IndexStorage::Handler::Pointer create(const std::string &, size_t)
    {
        return IndexStorage::Handler::Pointer();
    }

    //! Open a file
    virtual IndexStorage::Handler::Pointer open(const std::string &path, bool)
    {
        auto iter = _map.find(path);
        if (iter != _map.end()) {
            return IndexStorage::Handler::Pointer(
                new (std::nothrow) ExtraMemoryStorage::Handler(
                    iter->second.first, iter->second.second));
        }
        return IndexStorage::Handler::Pointer();
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

    //! Insert an extra memory
    bool emplace(const std::string &path, void *ptr, size_t len)
    {
        return _map.emplace(path, std::make_pair(ptr, len)).second;
    }

    //! Insert an extra memory
    bool emplace(std::string &&path, void *ptr, size_t len)
    {
        IndexStorage::Handler::Pointer handler(new Handler(ptr, len));
        return _map
            .emplace(std::forward<std::string>(path), std::make_pair(ptr, len))
            .second;
    }

    //! Erase an extra memory
    void erase(const std::string &path)
    {
        _map.erase(path);
    }

    //! Create an extra memory storage
    static ExtraMemoryStorage::Pointer Create(void)
    {
        return Pointer(new (std::nothrow) ExtraMemoryStorage());
    }

private:
    std::map<std::string, std::pair<void *, size_t>> _map;
};

} // namespace mercury

#endif // __MERCURY_EXTRA_MEMORY_STORAGE_H__
