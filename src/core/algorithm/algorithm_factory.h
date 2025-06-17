#ifndef __MERCURY_CORE_ALGORITHM_FACTORY_H__
#define __MERCURY_CORE_ALGORITHM_FACTORY_H__

#include "builder.h"
#include "index.h"
#include "merger.h"
#include "searcher.h"
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

class AlgorithmFactory {
public:
    AlgorithmFactory(const IndexParams& index_params)
        : index_params_(index_params) {
    }
    AlgorithmFactory() {}
    Builder::Pointer CreateBuilder();
    Searcher::Pointer CreateSearcher(bool in_mem = false);
    Merger::Pointer CreateMerger();
    Index::Pointer CreateIndex(bool for_load = false);
    void SetIndexParams(const IndexParams& index_params) {
        index_params_ = index_params;
    }

private:
    inline std::string GetAlgorithm() const {
        return index_params_.getString(PARAM_INDEX_TYPE);
    }
private:
    IndexParams index_params_;
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_ALGORITHM_FACTORY_H__
