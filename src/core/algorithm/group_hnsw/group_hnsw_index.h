#ifndef __MERCURY_CORE_GROUP_HNSW_INDEX_H__
#define __MERCURY_CORE_GROUP_HNSW_INDEX_H__

#include "src/core/framework/index_framework.h"
#include "src/core/utils/array_profile.h"
#include "src/core/utils/hash_table.h"
#include "src/core/common/params_define.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/index.h"
#include "src/core/algorithm/group_manager.h"
#include "src/core/utils/my_heap.h"
#include "coarse_hnsw_index.h"
#include "group_hnsw_info_manager.h"
#include "src/core/algorithm/pq_common.h"
#include <random>
#include <stdint.h>
#include <pthread.h>
#include "hnsw_context.h"
#include "visit_list.h"

MERCURY_NAMESPACE_BEGIN(core);

class GroupHnswIndex : public Index
{
public:
    GroupHnswIndex();

public:
    typedef std::shared_ptr<GroupHnswIndex> Pointer;
    virtual int Create(IndexParams& index_params);
    void SetBaseDocid(exdocid_t base_docid) override
    {
        base_docid_ = base_docid;
        coarse_hnsw_index_.SetBaseDocid(base_docid);
    }
    void SetVectorRetriever(const AttrRetriever& retriever) {
        coarse_hnsw_index_.SetVectorRetriever(retriever);
    }
    int CalCoarseIndexCapacity(const std::unordered_map<GroupInfo, uint32_t, GroupHnswHash, GroupHnswCmp>& group_meta);
    int AssignSpace(docid_t doc_id, const std::vector<GroupInfo>& group_infos,
                     std::vector<docid_t>& doc_ids,
                     std::vector<uint32_t>& doc_max_layers,
                     std::vector<uint32_t>& group_doc_ids,
                     std::vector<uint64_t>& group_offsets,
                     uint32_t doc_max_layer);
    const void * GetDocFeature(docid_t doc_id);
    int AddDoc(docid_t doc_id, uint64_t group_offset, const void *val, uint32_t doc_max_layer);
    int InitMappingSpace();
    int BaseIndexAdd(docid_t doc_id, pk_t pk, const void *val, size_t len);
    void RedundantMemClip();
    int Dump(const void*& data, size_t& size);
    void FreeMem();
    int Load(const void*, size_t) override;
    int KnnSearch(GroupInfo group_info, size_t topk, const void * query_val, size_t len, GeneralSearchContext*context, MyHeap<DistNode>* group_heap, int max_scan_num_in_query, std::pair<int, int>& cmp_cnt);
    int BruteSearch(GroupInfo group_info, size_t topk, const void * query_val, size_t len, GeneralSearchContext*context, std::vector<MyHeap<DistNode>>& group_heaps, std::pair<int, int>& cmp_cnt);
    size_t MemQuota2DocCount(size_t mem_quota, size_t elem_size) const override {
        //TODO
        return 0;
    }
    bool IsFull() const override {
        return GetDocNum() >= GetMaxDocNum(); 
    }
    int64_t UsedMemoryInCurrent() const override {
        //TODO
        return 0;
    }
    uint64_t GetMaxDocNum() const override {
        return group_max_doc_num_;
    }
    // no use
    float GetRankScore(docid_t doc_id) override { return 0;};
    size_t GetDocNum() const override {
        return group_doc_num_;
    }
    // bool InitHnswGroupInfo(const std::string& centroid_dir);
    // bool ResolveGroupFile(const std::string& meta_path, std::vector<GroupInfo>& groups, 
    //                       std::vector<std::pair<GroupInfo, GroupHnswInfo>>& group_hnsw_infos);
    uint32_t GetRandomLevel();

    // Dump index package without emplace back when merge
    const void* DumpInMerger(size_t& size) {
        const void* data = nullptr;
        if (!index_package_.dump(data, size)) {
            LOG_ERROR("Failed to dump package.");
        }
        return data;
    }

    CoarseHnswIndex& GetCoarseHnswIndex() {
        return coarse_hnsw_index_;
    }

private:
    static const int64_t MAX_LEVEL = 10;
    static const uint64_t MAX_SCALING_FACTOR = 128U;
    static const uint32_t MAX_NEIGHBOR_CNT = 255;

    //for random level
    std::mt19937 random_;

    //候选点数
    uint64_t ef_construction_;

    //最高层数
    uint32_t max_level_;

    //缩放系数，在HNSW图中，从概率上来说下面一层中doc数量是其上面一层doc数量的_scalingFactor倍
    uint64_t scaling_factor_;

    //其它层邻居数
    uint64_t upper_neighbor_cnt_;

    //第0层邻居数，为其它层邻居数的2倍
    uint64_t neighbor_cnt_;

    //配置中最大doc数量
    uint64_t group_max_doc_num_;
    
    //现有doc数量
    uint64_t group_doc_num_;

    //构建HNSW图doc数量阈值
    uint32_t build_threshold_;

    GroupManager group_manager_;

    GroupHnswManager group_hnsw_info_manager_;

    CoarseHnswIndex coarse_hnsw_index_;
    std::vector<char> coarse_hnsw_base_;

    //docid -> feature(vector)
    std::vector<uint8_t> feature_profile_base_;
    ArrayProfile feature_profile_; // store feature(vector) of all doc

    uint64_t part_dimension_;
    
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_GROUP_HNSW_INDEX_H__
