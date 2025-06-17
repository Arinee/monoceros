#include "index_faiss.h"
#include <iostream>
#include "faiss/index_io.h"
#include "storage_io_reader.h"

using namespace std;
using namespace mercury;

bool IndexFaiss::Load(IndexStorage::Handler::Pointer &&handler)
{
    StorageIOReader ioReader(std::move(handler));
    //TODO DO NOT support IO_FLAG_MMAP
    int ioFlags = 0;
    _index = faiss::read_index(&ioReader, ioFlags);
    return true;
}

bool IndexFaiss::Dump(IndexStorage::Pointer storage, const string& file_name, bool only_dump_meta)
{
    (void)(storage);(void)(file_name);(void)(only_dump_meta);
    std::cerr << "Not implemented yet..." << std::endl;
    return true;
}

bool IndexFaiss::Create(IndexStorage::Pointer storage, const string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle)
{
    (void)(storage);(void)(file_name);(void)(meta_file_handle);
    std::cerr << "Not implemented yet..." << std::endl;
    return true;
}

int IndexFaiss::Add(docid_t doc_id, uint64_t key, const void *val, size_t len)
{
    (void)(doc_id);(void)(key);(void)(val);(void)(len);
    std::cerr << "Not implemented yet..." << std::endl;
    return doc_id;
}

bool IndexFaiss::RemoveId(uint64_t key)
{
    (void)(key);
    std::cerr << "Not implemented yet..." << std::endl;
    return false;
}
