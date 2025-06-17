#ifndef __MERCURY_CORE_COARSE_HNSW_INDEX_H__
#define __MERCURY_CORE_COARSE_HNSW_INDEX_H__

// #include "hnsw_index.h"

#include "src/core/common/common.h"
#include "src/core/utils/array_profile.h"
#include "src/core/framework/index_error.h"
#include "src/core/algorithm/group_hnsw/visit_list.h"
#include "src/core/algorithm/group_hnsw/hnsw_context.h"
#include "src/core/framework/custom_distance/calculator_factory.h"
#include <vector>
#include <pthread.h>
#include <assert.h>
#include <limits>
#include <thread>
#include <mutex>

MERCURY_NAMESPACE_BEGIN(core);

class CoarseHnswIndex {
    
public:
    struct Header {
        uint64_t group_num:20;
        uint64_t capacity:44;
    };
    // 记录一个doc在某一层的邻居数量
    struct NeighborListHeader
    {
        uint8_t neighbor_cnt;//counter for neighbor list
        uint8_t pos;//start with 0, pos in neighbor list has statType state
        uint8_t state_type;//ordered or selected?
        uint8_t padding = 0xfe;
        static const uint8_t ORDERED_STATE = 0;
        static const uint8_t SELECTED_STATE = 1;
    };
    // 一个group元信息
    struct GroupHeader {
        uint64_t group_capacity;
        uint64_t neighbor_cur_offset;
        uint32_t doc_total_num;
        docid_t entry_point;
        uint32_t doc_cur_num;
        uint8_t cur_max_level;
        bool is_hnsw;
        uint8_t padding[2];
    };

public:
    CoarseHnswIndex();
    virtual ~CoarseHnswIndex();

public:

    bool create(void *pBase, size_t capacity, uint32_t group_num);

    int setGroupMeta(uint64_t group_offset, uint64_t group_end_offset, uint32_t group_doc_total_num, bool is_hnsw_group);
    void setIndexMeta(IndexMeta *index_meta, bool contain_feature) {
        index_meta_ = index_meta;
        contain_feature_ = contain_feature;
    };
    void SetBaseDocid(exdocid_t base_docid) {
        base_docid_ = base_docid;
    }
    void SetVectorRetriever(const AttrRetriever& retriever) {
        vector_retriever_ = retriever;
    }
    void setFeatureProfile(ArrayProfile *feature_profile) { feature_profile_ = feature_profile; }
    void setLevelOffset(uint32_t max_level, uint64_t upper_neighbor_cnt);
    void setSearchStep(int step) { step_ = step; }
    void setCandidateNums(uint32_t max_level, uint64_t ef_construction);
    void setMaxScanNums(int max_scan_num) { max_scan_num_ = max_scan_num; }
    int getMaxScanNums() {return max_scan_num_;}
    void setScorer(uint32_t part_dimension, CustomMethods custom_method);

    void addBruteGroupDoc(uint64_t group_offset, docid_t doc_id);
    int64_t addHnswGroupDocMeta(uint64_t group_offset, docid_t doc_id, uint32_t doc_max_layer);

    int addDoc(uint64_t group_offset, docid_t group_doc_id, const void *val, int32_t doc_max_layer);
    char * getBase() { return p_base_; }
    docid_t getGlobalDocId(char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id) {
        return *reinterpret_cast<docid_t *>(group_base + doc_offset_base[group_doc_id]);
    }

    void RedundantMemClip(std::vector<uint64_t*>& offsets);

    const void* GetBasePtr() const { return static_cast<const void*>(p_base_); }
    Header *getHeader() { return p_header_; }
    
    int load(void *pBase, size_t memory_size);
    
    int searchHnswNeighbors(uint64_t group_offset, TopkHeap &topk_heap, const void *query_val, size_t len, GeneralSearchContext* context, uint64_t &compare_cnt, int max_scan_num_in_query);
    int searchZeroHnswNeighbors(uint64_t group_offset, TopkHeap &topk_heap, const void *query_val, size_t len, uint64_t &compare_cnt, int max_scan_num_in_query);
    int searchBruteNeighbors(uint64_t group_offset, std::vector<DistNode> &dist_nodes, const void *query_val, size_t len, uint64_t &compare_cnt);
    int bruteSearchHnswNeighbors(uint64_t group_offset, std::vector<DistNode> &dist_nodes, const void *query_val, size_t len, uint64_t &compare_cnt);
    void printBruteGroupNeighbors(uint64_t group_offset) {
        GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(p_base_ + group_offset);
        if (pCurGroup->doc_total_num < 100)
            return;
        std::cout << "is hnsw group: " << pCurGroup->is_hnsw << std::endl;
        std::cout << "total doc num: " << pCurGroup->doc_total_num << std::endl;
        docid_t *doc_id_base = reinterpret_cast<docid_t *>(pCurGroup + 1);
        std::cout << "doc_id: ";
        for (uint32_t i = 0; i < 100 && i < pCurGroup->doc_total_num; i++) {
            docid_t docGlobalId = doc_id_base[i];
            std::cout << docGlobalId << " ";
        }
        std::cout << std::endl;
    }

private:
    
    int updateEntryPoint(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point,
                         float &dist, uint64_t &compare_cnt, GeneralSearchContext* context);
    int updateZeroEntryPoint(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point,
                         float &dist, uint64_t &compare_cnt);
    NeighborListHeader* getNeighborList(char* group_base, uint64_t *doc_offset_base, int32_t level, docid_t group_doc_id) {
        return reinterpret_cast<NeighborListHeader *>(group_base + doc_offset_base[group_doc_id] + level_offset_.at(level));
    }
    void addNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id, const void *val, 
                      int32_t level, docid_t &entry_point, float &dist, uint32_t group_doc_total_num, TopkHeap& level_topk_heap);
    //通过邻居的邻居扩展候选点集合
    int searchNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point, 
                         float &dist, TopkHeap &topk_heap, uint64_t &compare_cnt, uint32_t group_doc_total_num, GeneralSearchContext* context, int max_scan_num_in_query);
    int searchZeroNeighbors(char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point, 
                         float &dist, TopkHeap &topk_heap, uint64_t &compare_cnt, 
                         VisitList<uint32_t>& visit_list, CandidateHeap& candidates);
    //探索式算法从候选点集合选择最终邻居并写入，不存在竞争关系，无需加锁
    void selectNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id, int32_t level, TopkHeap &topk_heap);
    //更新邻居的邻居信息
    void updateNeighborLink(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t idx, int32_t level, TopkHeap& levelTopk);
    //更新某一层邻居的邻居信息，可能存在竞争关系，需要加锁
    void updateLink(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t main_node, docid_t link_node, int32_t level, float dist);

    int makeOrdered2Selected(char* group_base, uint64_t *doc_offset_base, docid_t main_node, NeighborListHeader *header, float *selected_dist);

    int selectLeftNeighbor(char* group_base, uint64_t *doc_offset_base, docid_t main_node, NeighborListHeader *header, int keep_size, int selected_pos, float *selected_dist);

private:

    char *p_base_;
    Header *p_header_;

    // Group自旋锁，保护groupHeader，根据GroupId计算取锁
    static const uint32_t GROUP_LOCK_NUMBER = 1U << 10;
    static const uint32_t GROUP_LOCK_MASK = (1U << 10) - 1U;
    std::mutex group_lock_[GROUP_LOCK_NUMBER];

    // 读写锁，写邻居信息时取锁，根据GroupOffset&InnerDocId计算取锁
    static const uint32_t LOCK_NUMBER = 1U << 16;
    static const uint32_t LOCK_MASK = (1U << 16) - 1U;
    //rwlock protecting GroupHeader
    std::mutex offset_lock_[LOCK_NUMBER];

    // 保存doc向量的ArrayProfile指针
    ArrayProfile *feature_profile_;
    AttrRetriever vector_retriever_;

    IndexMeta *index_meta_;
    bool contain_feature_ = false;

    CalculatorFactory calculator_factory_;
    IndexDistance::Measure measure_;

    uint64_t upper_neighbor_cnt_;
    uint64_t base_neighbor_cnt_;

    std::vector<uint64_t> level_offset_;

    int step_;

    std::vector<TopkHeap> level_topks_;

    int max_scan_num_;

    exdocid_t base_docid_;
};

MERCURY_NAMESPACE_END(core);
#endif //__MERCURY_CORE_COARSE_HNSW_INDEX_H__