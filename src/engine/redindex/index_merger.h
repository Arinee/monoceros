/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 14:53

#pragma once

#include "redindex_index.h"
#include "redindex_common.h"
#include "src/core/algorithm/merger.h"

namespace mercury {
namespace redindex {

class IndexMerger {
public:
    using Pointer = std::shared_ptr<IndexMerger>;

    IndexMerger(mercury::core::Merger::Pointer core_merger)
        : core_merger_(core_merger){
    }

    int PreUpdate(const std::vector<RedIndex::Pointer> &indexes,
                          const std::vector<RedIndexDocid> &new_redindex_docids);

    int Merge(const std::vector<RedIndex::Pointer> &indexes);

    const void * Dump(size_t *size);
    
    const void * DumpCentroid(size_t *size);

    size_t UsedMemoryInMerge();

private:
    std::vector<RedIndexDocid> new_redindex_docids_;
    mercury::core::Merger::Pointer core_merger_;
};

} // namespace redindex
} // namespace mercury
