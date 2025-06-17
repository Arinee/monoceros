/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-02 17:44

#include "index_searcher_factory.h"
#include "index_searcher.h"
#include "src/core/algorithm/algorithm_factory.h"

namespace mercury {
namespace redindex {

IndexSearcher::Pointer IndexSearcherFactory::Create(const SchemaParams& schema, bool in_mem)
{
    mercury::core::IndexParams index_params;
    SchemaToIndexParam(schema, index_params);
    mercury::core::AlgorithmFactory alg_factory(index_params);
    mercury::core::Searcher::Pointer searcher = alg_factory.CreateSearcher(in_mem);

    if (!searcher) {
        LOG_ERROR("create core searcher failed.");
        return IndexSearcher::Pointer();
    }
    return IndexSearcher::Pointer(new IndexSearcher(searcher));
}

} // namespace redindex
} // namespace mercury
