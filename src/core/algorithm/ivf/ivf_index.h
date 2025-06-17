#ifndef __MERCURY_CORE_IVF_INDEX_H__
#define __MERCURY_CORE_IVF_INDEX_H__

#include "../index.h"
#include "../centroid_resource.h"
#include "../coarse_index.h"

MERCURY_NAMESPACE_BEGIN(core);
typedef std::unique_ptr<char[]> MatrixPointer;

class IvfIndex : public Index {
public:
    typedef std::shared_ptr<IvfIndex> Pointer;
    /// load index from index package file and index will save this handle
    int Load(const void*, size_t) override;
    /// add a new vectoV
    virtual int Add(docid_t doc_id, pk_t pk,
                    const std::string& query_str, 
                    const std::string& primary_key = "") override;
    int Add(docid_t doc_id, pk_t pk, const void *val, size_t len);
    /// Display the actual class name and some more info
    void Display() const override{};
    /// whither index segment full
    bool IsFull() const override;

    float GetRankScore(docid_t doc_id) override;

    size_t MemQuota2DocCount(size_t memQuota, size_t elemSize) const override;

    size_t GetDocNum() const override {
        return coarse_index_.getUsedDocNum();
    }

    uint64_t GetMaxDocNum() const override {
        return coarse_index_.getHeader()->maxDocSize;
    }

    virtual int Create(IndexParams& index_params);

    virtual int Dump(const void*& data, size_t& size) override;

    int CopyInit(const Index*, size_t) override;

    int64_t UsedMemoryInCurrent() const override;

    SlotIndex GetNearestLabel(const void* data, size_t /* size */);

public:
    int AddBase(docid_t doc_id, pk_t pk);

    const CentroidResource& GetCentroidResource() const {
        return centroid_resource_;
    }

    CentroidResource& GetCentroidResource() {
        return centroid_resource_;
    }

    CoarseIndex<BigBlock>& GetCoarseIndex() {
        return coarse_index_;
    }

    ArrayProfile& GetSlotIndexProfile() {
        return slot_index_profile_;
    }

    std::string& GetRoughMatrix() {
        return rough_matrix_;
    }

    int SearchIvf(std::vector<CoarseIndex<BigBlock>::PostingIterator>& postings, const void* data, size_t size, const IndexParams& context_params = IndexParams());

protected:
    bool InitIvfCentroidMatrix(const std::string& centroid_dir);

    MatrixPointer DoLoadCentroidMatrix(const std::string& file_path, size_t dimension,
                                       size_t element_size, size_t& centroid_size) const;
    bool StrToValue(const std::string& source, void* value) const;

    size_t GetReserverdDocNum() {
        size_t reserve = index_params_.getUint64(PARAM_GENERAL_RESERVED_DOC);
        if (reserve == 0) {
            return DEFAULT_RESERVED_DOC;
        }

        return reserve;
    }

    int CompactEach(docid_t docid, IvfIndex* compact_index);

private:
    int CompactIndex();

private:
    CentroidResource centroid_resource_;
    CoarseIndex<BigBlock> coarse_index_;
    std::vector<char> coarse_base_;
    ArrayProfile slot_index_profile_; // docid to label
    std::vector<uint8_t> slot_index_profile_base_;

    std::string rough_matrix_;
    IvfIndex::Pointer compact_index_ = nullptr;

public:
    static constexpr size_t MIN_BUILD_COUNT = 100000;
};

MERCURY_NAMESPACE_END(core);

#endif
