/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-14 11:39

#include "index_merger_factory.h"
#include "redindex_index.h"
#include "index_merger.h"
#include "redindex_common.h"
#include "src/core/algorithm/algorithm_factory.h"

namespace mercury { 
namespace redindex {

IndexMerger::Pointer IndexMergerFactory::Create(const SchemaParams& schema)
{
    mercury::core::IndexParams index_params;
    SchemaToIndexParam(schema, index_params);
    if (schema.find(SCHEMA_MERGER_MEMORY_QUOTA) != schema.end()) {
        index_params.set(mercury::core::PARAM_GENERAL_INDEX_MEMORY_QUOTA, std::stol(schema.at(SCHEMA_MERGER_MEMORY_QUOTA)));
    }

    mercury::core::AlgorithmFactory alg_factory(index_params);
    mercury::core::Merger::Pointer merger = alg_factory.CreateMerger();
    if (!merger) {
        LOG_ERROR("create core merger failed.");
        return IndexMerger::Pointer();
    }
    return IndexMerger::Pointer(new IndexMerger(merger));
}

} // namespace redindex
} // namespace mercury
