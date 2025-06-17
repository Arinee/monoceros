/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-03 18:07

#pragma once

#include "redindex_index.h"
#include "index_builder.h"

namespace mercury { 
namespace redindex {

class IndexBuilderFactory {
public:
    static IndexBuilder::Pointer Create(const SchemaParams &schemaParams);
};

} // namespace redindex
} // namespace mercury
