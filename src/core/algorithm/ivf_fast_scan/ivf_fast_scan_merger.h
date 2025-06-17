#pragma once

#include "src/core/algorithm/merger.h"
#include "ivf_fast_scan_index.h"

MERCURY_NAMESPACE_BEGIN(core);

/*! Index Merger
 */
class IvfFastScanMerger : public Merger
{
public:
    //! Index Merger Pointer
    typedef std::shared_ptr<IvfFastScanMerger> Pointer;

    //! Initialize Merger
    int Init(const IndexParams &params) override;

    int MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) override;

    int MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) override;

    //! Dump index into file or memory
    int DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) override;

    const void * DumpIndex(size_t *size) override;

protected:
    bool CheckSize(const std::vector<Index::Pointer> &indexes, size_t new_size) const;

    bool PackFastScanCodes();

private:
    template <typename T>
    int DoMerge(const std::vector<Index::Pointer> &indexes,
                const std::vector<std::pair<size_t, T>> merge_map,
                bool by_pk = false);

    template <typename T>
    int DoMergeEachIvf(docid_t new_id, T old_id_pk, size_t index_offset,
                       const std::vector<Index::Pointer> &indexes, bool by_pk,
                       IvfFastScanIndex* merged_index);

    int GenerateSlotsProfile(const IvfFastScanIndex* ivf_index);

private:
    IvfFastScanIndex merged_index_;
    std::vector<std::vector<std::vector<SlotIndex>>> indexes_slots_profile_; //index->docid->slot indexes
};

MERCURY_NAMESPACE_END(core);
