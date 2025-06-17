#ifndef __MERCURY_CORE_GROUP_HNSW_BUILDER_H__
#define __MERCURY_CORE_GROUP_HNSW_BUILDER_H__

#include "src/core/framework/index_framework.h"
#include "src/core/algorithm/builder.h"
#include "group_hnsw_index.h"
#include <string>
#include <unordered_map>

MERCURY_NAMESPACE_BEGIN(core);

class GroupHnswBuilder : public Builder
{

public:
    //! Index Builder Pointer
    typedef std::shared_ptr<GroupHnswBuilder> Pointer;

    GroupHnswBuilder();

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

    Index::Pointer GetIndex() override {
        return index_;
    }

private:

    void DoAdd(docid_t global_doc_id, docid_t group_doc_id, uint64_t group_offset, const void *val, uint32_t doc_max_layer);
    float CalcScore(level_t level, group_t id);

private:
    GroupHnswIndex::Pointer  index_;
    ThreadPool pool_;
    int64_t built_cnt_;
    std::unordered_map<GroupInfo, uint32_t, GroupHnswHash, GroupHnswCmp> group_meta_;
    std::vector<std::pair<docid_t, std::vector<GroupInfo>>> doc_infos_;
    uint32_t sort_build_group_level_;
};


MERCURY_NAMESPACE_END(core);
#endif //__MERCURY_CORE_GROUP_HNSW_BUILDER_H__
