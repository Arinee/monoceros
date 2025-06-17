#ifndef __MERCURY_INDEX_FAISS_H__
#define __MERCURY_INDEX_FAISS_H__

#include "index/index.h"
#include "index/general_search_context.h"
#include "faiss/Index.h"

namespace mercury {

class IndexFaiss : public Index
{
public:
    typedef std::shared_ptr<IndexFaiss> Pointer;

    void UnLoad() override { std::cerr << "Not implemented yet..." << std::endl; }
    bool Load(IndexStorage::Handler::Pointer &&file_handle) override;
    bool Dump(IndexStorage::Pointer storage, const std::string& file_name, bool only_dump_meta) override;
    bool Create(IndexStorage::Pointer storage, 
            const std::string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle) override;
    int Add(docid_t doc_id, uint64_t key, const void *val, size_t len) override;
    bool RemoveId(uint64_t key) override;
    int Search(const void* x, size_t k , void* dist, void* label) {
        _index->search(1,static_cast<const float*>(x), k,
                static_cast<float*>(dist), static_cast<long*>(label));
        return 0;
    }

    GENERATE_RETURN_EMPTY_INDEX(IndexFaiss);
protected:
    faiss::Index* _index;
};

} // namespace mercury

#endif // __MERCURY_INDEX_FAISS_H__
