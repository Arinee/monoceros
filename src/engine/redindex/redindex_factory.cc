/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-13 17:44

#include "redindex_factory.h"
#include "redindex_index.h"
#include "src/core/algorithm/algorithm_factory.h"

namespace mercury {
namespace redindex {

RedIndex::Pointer RedIndexFactory::Create(const SchemaParams& schema)
{
    mercury::core::IndexParams index_params;
    SchemaToIndexParam(schema, index_params);
    mercury::core::AlgorithmFactory alg_factory(index_params);
    mercury::core::Index::Pointer core_index = alg_factory.CreateIndex();
    if (!core_index) {
        LOG_ERROR("create core index failed.");
        return RedIndex::Pointer();
    }
    return RedIndex::Pointer(new RedIndex(core_index));
}

RedIndex::Pointer RedIndexFactory::Load(const SchemaParams& schema,
                                                const void *data, size_t size)
{
    mercury::core::IndexParams index_params;
    SchemaToIndexParam(schema, index_params);
    mercury::core::AlgorithmFactory alg_factory(index_params);
    mercury::core::Index::Pointer core_index = alg_factory.CreateIndex(true);

    if (!core_index) {
        LOG_ERROR("create core index failed.");
        return RedIndex::Pointer();
    }

    RedIndex::Pointer redindex_index = RedIndex::Pointer(new RedIndex(core_index));

    if (redindex_index->Load(data, size) != 0) {
        std::cerr << "Failed to load index." << std::endl;
        return RedIndex::Pointer();
    }

    return redindex_index;
}

} // namespace redindex
} // namespace mercury
