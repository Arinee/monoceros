#pragma once

#include "ivf_rpq_index.h"
#include "src/core/algorithm/builder.h"
#include "src/core/common/common.h"
#include "src/core/framework/index_storage.h"
#include <memory>

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Builder
 */

class IvfRpqBuilder : public Builder
{
public:
    //! Index Builder Pointer
    typedef std::shared_ptr<IvfRpqBuilder> Pointer;

    IvfRpqBuilder() {}

    //! Initialize Builder
    int Init(IndexParams &index_params) override;
    //! Build the index
    int AddDoc(docid_t doc_id, uint64_t pk, const std::string &build_str, const std::string &primary_key = "") override;

    int GetRankScore(const std::string &build_str, float *score) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    const void *DumpIndex(size_t *size) override;

    Index::Pointer GetIndex() override
    {
        return index_;
    }

private:
    float CalcScore(uint32_t level, SlotIndex label);

private:
    IndexParams index_params_;
    uint32_t sort_build_group_level_;
    IvfRpqIndex::Pointer index_;
};

MERCURY_NAMESPACE_END(core);
