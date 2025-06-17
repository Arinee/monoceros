/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 11:04

#pragma once

#include "index_searcher.h"

namespace mercury { 
namespace redindex {

class IndexSearcherFactory {
public:
    static IndexSearcher::Pointer Create(const SchemaParams& schemaParams, bool in_mem = false);
};

} // namespace redindex
} // namespace mercury
