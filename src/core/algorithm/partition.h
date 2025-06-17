#ifndef __MERCURY_PARTITION_H__
#define __MERCURY_PARTITION_H__

#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

class Partition
{  
public:
    size_t node_start_ = 0;
    size_t posting_start_ = 0;
    size_t posting_end_ = 0;

    Partition(size_t node_start, size_t posting_start, size_t posting_end) {
        node_start_ = node_start;
        posting_start_ = posting_start;
        posting_end_ = posting_end;
    }
    Partition(const Partition& rhs) {
        node_start_ = rhs.node_start_;
        posting_start_ = rhs.posting_start_;
        posting_end_ = rhs.posting_end_;
    }

    ~Partition() {}
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_PARTITION_H__