/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-09-03 11:04

#pragma once

#include "redindex_index.h"

namespace mercury { 
namespace redindex {

class RedIndexFactory {
public:
    static RedIndex::Pointer Create(const SchemaParams& schemaParams);
    static RedIndex::Pointer Load(const SchemaParams& schemaParams, 
                                      const void *data, size_t size);
};

} // namespace redindex
} // namespace mercury
