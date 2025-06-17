#ifndef __MERCURY_CORE_IVF_PQ_MERGER_H__
#define __MERCURY_CORE_IVF_PQ_MERGER_H__

#include "../ivf/ivf_merger.h"
#include "ivf_pq_index.h"

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Merger
 */
class IvfPqMerger : public IvfMerger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<IvfPqMerger> Pointer;

    //! Initialize Merger
    int Init(const IndexParams &params) override;

    int MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) override;

    int MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    const void * DumpIndex(size_t *size) override;

private:
    template <typename T>
    int DoMerge(const std::vector<Index::Pointer> &indexes,
                const std::vector<std::pair<size_t, T>> merge_map,
                bool by_pk = false);

    template <typename T>
    int DoMergeEachPq(docid_t new_id, T old_id_pk, size_t index_offset,
                      const std::vector<Index::Pointer> &indexes, bool by_pk);

private:
    IvfPqIndex merged_index_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_IVF_PQ_MERGER_H__
