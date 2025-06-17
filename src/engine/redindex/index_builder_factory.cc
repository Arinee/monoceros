/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-13 15:00

#include "index_builder_factory.h"
#include "redindex_common.h"
#include "src/core/framework/index_params.h"
#include "src/core/algorithm/algorithm_factory.h"

namespace mercury { 
namespace redindex {

IndexBuilder::Pointer IndexBuilderFactory::Create(const SchemaParams& schema)
{
    mercury::core::IndexParams index_params;
    SchemaToIndexParam(schema, index_params);
    if (schema.find(SCHEMA_BUILDER_MEMORY_QUOTA) != schema.end()) {
        index_params.set(mercury::core::PARAM_GENERAL_INDEX_MEMORY_QUOTA, std::stol(schema.at(SCHEMA_BUILDER_MEMORY_QUOTA)));
    }
    mercury::core::AlgorithmFactory alg_factory(index_params);
    mercury::core::Builder::Pointer builder = alg_factory.CreateBuilder();

    if (!builder) {
        LOG_ERROR("create core builder failed.");
        return IndexBuilder::Pointer();
    }

    return IndexBuilder::Pointer(new IndexBuilder(builder));
}

} // namespace redindex
} // namespace mercury
