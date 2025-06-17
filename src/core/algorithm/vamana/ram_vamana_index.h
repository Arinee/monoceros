# pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/index.h"
#include "coarse_vamana_index.h"
#include "src/core/algorithm/centroid_resource.h"
#include "src/core/algorithm/centroid_resource_manager.h"
#include "src/core/utils/vamana/aligned_file_reader.h"
#include "src/core/utils/vamana/percentile_stats.h"
#include "src/core/algorithm/query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

class RamVamanaIndex : public Index
{
public:
    RamVamanaIndex();

    ~RamVamanaIndex();

    typedef std::shared_ptr<RamVamanaIndex> Pointer;

    virtual int Create(IndexParams& index_params);

    int Load(const void* data, size_t size) override;

    size_t MemQuota2DocCount(size_t mem_quota, size_t elem_size) const override {
        //TODO
        return 0;
    }

    bool IsFull() const override {
        return false; 
    }

    int64_t UsedMemoryInCurrent() const override {
        //TODO
        return 0;
    }

    uint64_t GetMaxDocNum() const override {
        return max_doc_num_;
    }

    size_t GetDocNum() const override {
        return 0;
    }

    // no use
    float GetRankScore(docid_t doc_id) override { 
        return 0; 
    }

    int Dump(const void*& data, size_t& size) override;

    int BaseIndexAdd(docid_t doc_id, const void *val);

    void BuildMemIndex();

    void DumpMemLocal(const std::string& ram_index_path);

    void LoadMemLocal(const std::string& ram_index_path);

    void Search(const void *query, size_t K, uint32_t L, uint64_t *indices,
                float *distances, uint32_t &num_cmps);

    void GetBaseVec(docid_t doc_id, void *dest);

    size_t GetBaseVecNum();

    uint32_t GetL();

private:
    // (default is 64): the degree of the graph index, typically between 60 and 150. 
    // Larger R will result in larger indices and longer indexing times, but better search quality. 
    uint32_t R_;

    // (default is 100): the size of search list during index build. 
    // Typical values are between 75 to 200. 
    // Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. 
    // Use a value for L value that is at least the value of R unless you need to build indices really quickly and can somewhat compromise on quality. 
    uint32_t L_;

    // (default is 1): num of threads to build coarse vamana index
    uint32_t T_;

    // doc num for search
    uint64_t search_doc_num_;

    // doc num for build
    uint64_t build_doc_num_;

    // base Vamana index structure
    std::unique_ptr<CoarseVamanaIndex> coarseVamanaIndex_;

    bool _use_half;

    uint64_t data_dim_ = 0;

    uint16_t data_size_ = 0;

};

MERCURY_NAMESPACE_END(core);