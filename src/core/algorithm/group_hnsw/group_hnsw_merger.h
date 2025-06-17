#ifndef __MERCURY_CORE_GROUP_HNSW_MERGER_H__
#define __MERCURY_CORE_GROUP_HNSW_MERGER_H__

#include "../merger.h"
#include "group_hnsw_index.h"

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Merger
 */
class GroupHnswMerger : public Merger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<GroupHnswMerger> Pointer;

    //! Initialize Merger
    int Init(const IndexParams &params) override;

    int MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) override;

    int MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    const void * DumpIndex(size_t *size) override;

private:
    GroupHnswIndex* merged_index_;
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_GROUP_HNSW_MERGER_H__
