#include "partition_strategy.h"

MERCURY_NAMESPACE_BEGIN(core);

PartitionStrategy::PartitionStrategy(size_t doc_num, size_t doc_num_per_concurrency, size_t max_concurrency_num, bool need_parallel) {
    concurrency_num_ = std::min(max_concurrency_num, (size_t)ceil(doc_num / (double_t)doc_num_per_concurrency));
    if (!need_parallel || concurrency_num_ <= 0) {
        concurrency_num_ = 1;
    }
    //LOG_INFO("PartitionStrategy Info: need_parallel: %d, concurrency_num: %d, max_concurrency_num: %d, doc_num_per_concurrency: %d, doc_num: %d", need_parallel, concurrency_num_, max_concurrency_num, doc_num_per_concurrency, doc_num);
}

PartitionStrategy::~PartitionStrategy() {}

int PartitionStrategy::MixedPartitionByPosting(std::vector<size_t>& ivf_postings_info,
                                    PartitionStrategy::return_type &result) const {
    //support pure group partition
    size_t node_start = 0;
    size_t posting_start = 0;
    for (size_t i = 0; i < concurrency_num_ - 1; i++) {
        size_t posting_end = 0;
        posting_end = posting_start + ivf_postings_info.size() / concurrency_num_;
        result.emplace_back(node_start, posting_start, posting_end);
        
        //for next partition
        node_start += GetPostingsNodeCount(ivf_postings_info, posting_start, posting_end);
        posting_start = posting_end;
    }
    result.emplace_back(node_start, posting_start, ivf_postings_info.size());

    return 0;
}

int PartitionStrategy::MixedPartitionByDoc(size_t doc_num,
                                           PartitionStrategy::return_type &result) const {
    //support pure group partition by doc
    size_t node_start = 0;
    size_t posting_start = 0;
    for (size_t i = 0; i < concurrency_num_ - 1; i++) {
        size_t posting_end = 0;
        posting_end = posting_start + doc_num / concurrency_num_;
        result.emplace_back(node_start, posting_start, posting_end);
        
        //for next partition
        node_start = posting_start = posting_end;
    }
    result.emplace_back(node_start, posting_start, doc_num);

    return 0;
}



size_t PartitionStrategy::GetPostingsNodeCount(const std::vector<size_t>& ivf_postings_info,
                                              size_t start, size_t end) const {
    size_t count = 0;
    for (size_t i = start; i < end; i++) {
        count += ivf_postings_info.at(i);
    }

    return count;
}

MERCURY_NAMESPACE_END(core);
