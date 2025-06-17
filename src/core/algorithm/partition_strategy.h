#ifndef __MERCURY_PARTITION_STRATEGY_H__
#define __MERCURY_PARTITION_STRATEGY_H__

//#include "coarse_index.h"
#include <vector>
#include <memory>
#include <utility>
#include <sys/stat.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <cmath>

#include "query_info.h"
#include "partition.h"
#include "src/core/algorithm/thread_common.h"

MERCURY_NAMESPACE_BEGIN(core);

class PartitionStrategy
{
public:
    typedef std::vector<Partition> return_type;

    PartitionStrategy(size_t posting_num, size_t doc_num_per_concurrency, size_t max_concurrency_num, bool need_parallel);
    ~PartitionStrategy();

    int MixedPartitionByPosting(std::vector<size_t>& ivf_postings,
                                PartitionStrategy::return_type &result) const;
    int MixedPartitionByDoc(size_t doc_num,
                            PartitionStrategy::return_type &result) const;

private:
    size_t GetPostingsNodeCount(const std::vector<size_t>& ivf_postings, size_t start, size_t end) const;

private:
    size_t concurrency_num_; //计算得到的并发数

};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_PARTITION_STRATEGY_H__
