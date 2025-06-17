/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     gpu_group_ivf_batch_task.h
 *   \author   anduo@xiaohongshu.com
 *   \date     July 2022
 *   \version  1.0.0
 *   \brief    gpu group ivf batch task
 */

#include "bthread/bthread.h"
#include "putil/mem_pool/Pool.h"
#include "src/core/algorithm/query_info.h"
#include "src/core/utils/batching_util/batch_scheduler.h"

MERCURY_NAMESPACE_BEGIN(core);

class GpuGroupIvfBatchTask : public BatchTask
{
public:
    GpuGroupIvfBatchTask(std::vector<uint32_t> *dist_nodes, const QueryInfo *query_info, uint32_t total_topk,
                         uint32_t group_num, std::vector<uint32_t> *group_doc_num, char **return_data, bool enable_cate,
                         bool enable_rt, putil::mem_pool::Pool *pool, bthread_cond_t *batch_cond,
                         bthread_mutex_t *batch_mutex)
        : dist_nodes_(dist_nodes), query_info_(query_info), total_topk_(total_topk), group_num_(group_num),
          group_doc_nums_(group_doc_num), return_data_(return_data), enable_cate_(enable_cate), enable_rt_(enable_rt),
          pool_(pool), batch_cond_(batch_cond), batch_mutex_(batch_mutex), is_finish(false), is_success(false),
          is_destroy(false), size_(1){};
    GpuGroupIvfBatchTask() = delete;

    virtual size_t size() const override
    {
        return size_;
    }

public:
    // gpu计算需要变量
    std::vector<uint32_t> *dist_nodes_;
    const QueryInfo *query_info_;
    uint32_t total_topk_;
    uint32_t group_num_;
    std::vector<uint32_t> *group_doc_nums_;
    char **return_data_;
    bool enable_cate_;
    bool enable_rt_;
    putil::mem_pool::Pool *pool_;

    bthread_cond_t *batch_cond_;
    bthread_mutex_t *batch_mutex_;

    bool is_finish;
    bool is_success;
    bool is_destroy;

private:
    // TODO: 按doc_num或者group_num来控制batch大小
    // 目前是按query的数量当做size，所以默认一个task是1
    size_t size_;
};

MERCURY_NAMESPACE_END(core);
