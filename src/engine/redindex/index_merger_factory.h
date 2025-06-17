/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-08-29 14:53

#pragma once

#include "index_merger.h"
#include "redindex_common.h"

namespace mercury { 
namespace redindex {

class IndexMergerFactory {
public:
    static IndexMerger::Pointer Create(const SchemaParams &schemaParams);
};

} // namespace redindex
} // namespace mercury
