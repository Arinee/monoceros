#ifndef __MERCURY_CORE_IVF_MERGER_H__
#define __MERCURY_CORE_IVF_MERGER_H__

#include "../merger.h"
#include "ivf_index.h"

MERCURY_NAMESPACE_BEGIN(core);

template <typename T>
int DoMergeEachIvf(docid_t new_id, T old_id_pk, size_t index_offset,
                   const std::vector<Index::Pointer> &indexes, bool by_pk,
                   IvfIndex* merged_index);

/*! Index Merger
 */
class IvfMerger : public Merger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<IvfMerger> Pointer;

    //! Initialize Merger
    int Init(const IndexParams &params) override;

    int MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) override;

    int MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    const void * DumpIndex(size_t *size) override;

protected:
    bool CheckSize(const std::vector<Index::Pointer> &indexes, size_t new_size) const;

private:
    template <typename T>
    int DoMerge(const std::vector<Index::Pointer> &indexes,
                const std::vector<std::pair<size_t, T>> merge_map,
                bool by_pk = false);

private:
    IvfIndex merged_index_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_MERGER_H__
