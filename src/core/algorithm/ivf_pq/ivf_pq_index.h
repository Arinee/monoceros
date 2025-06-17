#ifndef __MERCURY_CORE_IVF_PQ_INDEX_H__
#define __MERCURY_CORE_IVF_PQ_INDEX_H__

#include "../ivf/ivf_index.h"
#include "src/core/framework/index_logger.h"

MERCURY_NAMESPACE_BEGIN(core);

class IvfPqIndex : public IvfIndex {
public:
    typedef std::shared_ptr<IvfPqIndex> Pointer;
    /// load index from index package file and index will save this handle
    int Load(const void*, size_t) override;
    /// add a new vectoV
    virtual int Add(docid_t doc_id, pk_t pk,
                    const std::string& query_str, 
                    const std::string& primary_key = "") override;
    int Add(docid_t doc_id, pk_t pk, const void *val, size_t len);
    /// Display the actual class name and some more info
    void Display() const override{};

    size_t MemQuota2DocCount(size_t memQuota, size_t elemSize) const override;

    int Create(IndexParams& index_params) override;
    virtual int Dump(const void*& data, size_t& size) override;

    int CopyInit(const Index*, size_t) override;

    int64_t UsedMemoryInCurrent() const override;

//public:
    //int AddBase(docid_t doc_id, pk_t pk);

public:
    ArrayProfile& GetPqCodeProfile() {
        return pq_code_profile_;
    }

    std::string& GetIntegrateMatrixStr() {
        return integrate_matrix_str_;
    }

private:
    bool InitPqCentroidMatrix(const IndexParams& param);
    int CompactIndex();

private:
    ArrayProfile pq_code_profile_; // doc1 对应的32个pq中心点的index
    std::vector<char> pq_code_base_;
    std::string integrate_matrix_str_;
    IvfPqIndex::Pointer compact_index_ = nullptr;
};

MERCURY_NAMESPACE_END(core);

#endif
