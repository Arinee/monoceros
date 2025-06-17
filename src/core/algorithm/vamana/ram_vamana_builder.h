# pragma once

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/builder.h"
#include "ram_vamana_index.h"
#include "bthread/bthread.h"
#include "src/core/algorithm/query_info.h"
#include <omp.h>

MERCURY_NAMESPACE_BEGIN(core);

class RamVamanaBuilder : public Builder
{

public:
    //! Index Builder Pointer
    typedef std::shared_ptr<RamVamanaBuilder> Pointer;

    RamVamanaBuilder();

    //! Initialize Builder
    int Init(IndexParams &params) override;

    //! Build the index
    int AddDoc(docid_t doc_id, uint64_t pk,
               const std::string& build_str, 
               const std::string& primary_key = "") override;

    int GetRankScore(const std::string& build_str, float * score) override;

    //! Dump index into file
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    //! Dump index into memory
    const void * DumpIndex(size_t* size) override;

    // Dump ram vamana index
    int DumpRamVamanaIndex(std::string &path_prefix) override;

    Index::Pointer GetIndex() override {
        return index_;
    }

private:
    RamVamanaIndex::Pointer index_;
};


MERCURY_NAMESPACE_END(core);
