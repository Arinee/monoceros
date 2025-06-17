#ifndef __MERCURY_FAISS_SERVICE_H__
#define __MERCURY_FAISS_SERVICE_H__

#include "framework/index_params.h"
#include "framework/search_result.h"
#include "index/base_index_provider.h"
#include "framework/index_params.h"
#include <memory>
#include <string>
#include <vector>

namespace mercury 
{

class FaissService : public VectorService
{
public:
    //! constructor
    FaissService(void)
        :_indexProvider(nullptr)
    {}

    //! Destructor
    ~FaissService(void) override;

    //! Initialize
    int Init(const IndexParams &params) override;

    //! Cleanup
    int Cleanup(void) override;

    //! Load index from file path or dir
    int LoadIndex(const std::string &prefix,
                          const IndexStorage::Pointer &stg) override;

    //! Unload index
    int UnloadIndex(void) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    //! Create a context
    SearchContext::Pointer CreateContext(const IndexParams &params) override;

    //! KNN Search
    int KnnSearch(size_t topk, const void *val, size_t len,
                          SearchContext::Pointer &context) override;

    //! KNN Exhaustive Search
    int ExhaustiveSearch(size_t topk, const void *val,
                                 size_t len, SearchContext::Pointer &context) override;

    //! Add a vector into index, 0 indicates success
    int AddVector(uint64_t key, const void * val, size_t len) override;

    //! Delete a vector from index
    int DeleteVector(uint64_t key) override;

    //! Update a vector in index
    int UpdateVector(uint64_t key, const void * val, size_t len) override;

protected:
    FaissIndexProvider *_indexProvider = nullptr;
    IndexMeta _indexMeta;
    IndexParams::Pointer _defaultParams;
};



} // namespace mercury

#endif // __MERCURY_FAISS_SERVICE_H__

