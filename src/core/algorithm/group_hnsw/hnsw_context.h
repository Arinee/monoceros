#ifndef __MERCURY_CORE_HNSW_CONTEXT_H__
#define __MERCURY_CORE_HNSW_CONTEXT_H__

#include "src/core/common/common_define.h"
#include "src/core/utils/heap.h"
#include "src/core/algorithm/general_search_context.h"
#include "src/core/algorithm/group_hnsw/visit_list.h"
#include <atomic>

MERCURY_NAMESPACE_BEGIN(core);

struct NodeSearchContext
{
    NodeSearchContext(void)
        : neighborList(nullptr),
          idx(-1),
          curPos(-1)
    {
    }

    NodeSearchContext(docid_t *nodeNeighborList, docid_t nodeIdx, int nodePos)
        : neighborList(nodeNeighborList),
          idx(nodeIdx),
          curPos(nodePos)
    {
    }

    docid_t *neighborList;
    docid_t idx;
    //current position in neighbor list
    int curPos;
};

struct GraphStat
{
    std::vector<uint64_t> levelVisitedCnt;
};

class CandidateHeap
{
public:
    CandidateHeap(void)
    : cur_size_(0),
      capacity_(0),
      ctx_arr_(nullptr)
    {
    }

    CandidateHeap(int k)
    : heap_(k)
    {
        cur_size_ = 0;
        capacity_ = k;
        ctx_arr_ = new (std::nothrow) NodeSearchContext[k];
    }

    ~CandidateHeap(void)
    {
        delete [] ctx_arr_;
        ctx_arr_ = nullptr;
    }

public:
    inline void top(NodeSearchContext *&key, float &value)
    {
        heap_.top(key, value);
    }

    inline float topValue(void)
    {
        return heap_.topValue();
    }

    inline void push(const NodeSearchContext &key, const float value)
    {
        if (cur_size_ >= capacity_) {
            return;
        }

        NodeSearchContext *ctx = ctx_arr_ + cur_size_;
        *ctx = key;
        heap_.push(ctx, value);
        cur_size_++;

        return;
    }

    inline void emplace_push(docid_t *nodeNeighborList, docid_t nodeIdx, 
                             int nodePos, const float value)
    {
        if (cur_size_ >= capacity_) {
            return;
        }

        NodeSearchContext *ctx = ctx_arr_ + cur_size_;
        ctx->neighborList = nodeNeighborList;
        ctx->idx = nodeIdx;
        ctx->curPos = nodePos;
        heap_.push(ctx, value);
        cur_size_++;

        return;
    }

    inline void pop(void)
    {
        heap_.pop();
    }

    inline void reset(int k = -1)
    {
        heap_.reset(k);

        cur_size_ = 0;

        if (k == -1) {
            return;
        }
        
        if (k > capacity_) {
            delete [] ctx_arr_;
            ctx_arr_ = new (std::nothrow) NodeSearchContext[k];
            capacity_ = k;
        }
    }

    inline bool empty(void)
    {
        return heap_.empty();
    }

public:
    GraphStat stat_;

private:
    Heap<NodeSearchContext *, float, std::greater<float> > heap_;
    int cur_size_;
    int capacity_;
    NodeSearchContext *ctx_arr_;
};

using TopkHeap = Heap<docid_t, float>;

MERCURY_NAMESPACE_END(core);
#endif //__MERCURY_CORE_HNSW_CONTEXT_H__
