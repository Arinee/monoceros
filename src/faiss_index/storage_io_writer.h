//
// Created by 吴文杰 on 2019-06-30.
//

#ifndef MERCURY_PROJECT_STORAGE_IO_WRITER_H
#define MERCURY_PROJECT_STORAGE_IO_WRITER_H

#include <stdio.h>
#include "faiss/Index.h"
#include "framework/index_storage.h"

namespace mercury {

class StorageIOWriter: public faiss::IOWriter {
public:
    StorageIOWriter(IndexStorage::Handler::Pointer &handler)
    : _handler(std::move(handler))
    {};
    ~StorageIOWriter() override = default;

    size_t operator()(
            const void *ptr, size_t size, size_t nitems) override {
        size_t count =  _handler->write(ptr, size * nitems);
        return count / size;
    }

    //TODO: don't support return fileno
    int fileno() override {
        return -1;
    }

private:
    IndexStorage::Handler::Pointer _handler;
};

}; // namespace mercury


#endif //MERCURY_PROJECT_STORAGE_IO_WRITER_H
