//
// Created by 吴文杰 on 2019-06-30.
//

#ifndef MERCURY_PROJECT_STORAGE_IO_READER_H
#define MERCURY_PROJECT_STORAGE_IO_READER_H

#include "faiss/AuxIndexStructures.h"
#include "framework/index_storage.h"

namespace mercury {

class StorageIOReader: public faiss::IOReader {
public:
    StorageIOReader(IndexStorage::Handler::Pointer &&handler)
    : _handler(std::move(handler))
    {};
    ~StorageIOReader() override = default;

    size_t operator()(void *ptr, size_t size, size_t nitems) override {
        size_t count =  _handler->read(ptr, size * nitems);
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


#endif //MERCURY_PROJECT_STORAGE_IO_READER_H
