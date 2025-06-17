/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     file_storage.cc
 *   \author   Hechong.xyf
 *   \date     Mar 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury File Storage
 */

#include "instance_factory.h"
#include "utility/file.h"
#include <vector>

namespace mercury {

/*! File Storage
 */
struct FileStorage : public IndexStorage
{
    /*! File Handler
     */
    class Handler : public IndexStorage::Handler
    {
    public:
        //! Constructor
        Handler(void) : _file() {}

        //! Destructor
        ~Handler(void)
        {
            this->cleanup();
        }

        //! Write data into the storage
        virtual size_t write(const void *data, size_t len)
        {
            return _file.write(data, len);
        }

        //! Write data into the storage
        virtual size_t write(size_t offset, const void *data, size_t len)
        {
            return _file.write(offset, data, len);
        }

        //! Read data from the storage (Zero-copy)
        virtual size_t read(const void **data, size_t len);

        //! Read data from the storage (Zero-copy)
        virtual size_t read(size_t offset, const void **data, size_t len);

        //! Read data from the storage
        virtual size_t read(void *data, size_t len)
        {
            return _file.read(data, len);
        }

        //! Read data from the storage
        virtual size_t read(size_t offset, void *data, size_t len)
        {
            return _file.read(offset, data, len);
        }

        //! Close the writer
        virtual void close(void)
        {
            this->cleanup();
            _file.close();
        }

        //! Reset the hanlder
        virtual void reset(void)
        {
            this->cleanup();
            _file.reset();
        }

        //! Retrieve size of file
        virtual size_t size(void) const
        {
            return _file.size();
        }

        //! Clean up
        void cleanup(void)
        {
            for (auto iter = _buf_vec.begin(); iter != _buf_vec.end(); ++iter) {
                delete[](*iter);
            }
            _buf_vec.clear();
        }

        //! Create a file
        bool create(const std::string &path, size_t file_size)
        {
            size_t last_slash = path.rfind('/');
            if (last_slash != std::string::npos) {
                File::MakePath(path.substr(0, last_slash).c_str());
            }
            return _file.create(path.c_str(), file_size);
        }

        //! Open a file
        bool open(const std::string &path, bool rdonly)
        {
            return _file.open(path.c_str(), rdonly);
        }

    private:
        File _file;
        std::vector<uint8_t *> _buf_vec;
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

size_t FileStorage::Handler::read(const void **data, size_t len)
{
    uint8_t *buf = new (std::nothrow) uint8_t[len];
    if (buf) {
        _buf_vec.push_back(buf);
        *data = buf;
        return _file.read(buf, len);
    }
    return 0u;
}

size_t FileStorage::Handler::read(size_t offset, const void **data, size_t len)
{
    uint8_t *buf = new (std::nothrow) uint8_t[len];
    if (buf) {
        _buf_vec.push_back(buf);
        *data = buf;
        return _file.read(offset, buf, len);
    }
    return 0u;
}

IndexStorage::Handler::Pointer FileStorage::create(const std::string &path,
                                                   size_t size)
{
    FileStorage::Handler *handler = new (std::nothrow) Handler();
    if (handler) {
        if (!handler->create(path, size)) {
            delete handler;
            handler = nullptr;
        }
    }
    return IndexStorage::Handler::Pointer(handler);
}

IndexStorage::Handler::Pointer FileStorage::open(const std::string &path,
                                                 bool rdonly)
{
    FileStorage::Handler *handler = new (std::nothrow) Handler();
    if (handler) {
        if (!handler->open(path, rdonly)) {
            delete handler;
            handler = nullptr;
        }
    }
    return IndexStorage::Handler::Pointer(handler);
}

INSTANCE_FACTORY_REGISTER_STORAGE(FileStorage);

} // namespace mercury
