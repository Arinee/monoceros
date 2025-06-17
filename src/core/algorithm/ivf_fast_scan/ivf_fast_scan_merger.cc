#include "ivf_fast_scan_merger.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Merger
int IvfFastScanMerger::Init(const IndexParams &params) {
    merged_index_.SetIndexParams(params);
    return 0;
}

template <typename T>
int IvfFastScanMerger::DoMergeEachIvf(docid_t new_id, T old_id_pk, size_t index_offset,
                                    const std::vector<Index::Pointer> &indexes, bool by_pk,
                                    IvfFastScanIndex* merged_index) {
    if (merged_index == nullptr) {
        LOG_ERROR("merged_index is null");
        return -1;
    }

    pk_t pk = old_id_pk;
    docid_t old_id = old_id_pk;

    assert(index_offset != INVALID_DOC_ID);
    auto ivf_index = dynamic_cast<IvfFastScanIndex*>(indexes.at(index_offset).get());
    if (by_pk) {
        if (!ivf_index->WithPk()) {
            LOG_ERROR("index has no id_map");
            return -1;
        }

        HashTable<pk_t, docid_t>& id_map = ivf_index->GetIdMap();
        if (!id_map.find(pk, old_id)) {
            LOG_ERROR("get docid from pk failed. key: %lu", pk);
            return -1;
        }
    }

    merged_index->AddBase(new_id, pk);
    if (merged_index->ContainFeature()) {
        merged_index->GetFeatureProfile().insert(new_id, ivf_index->GetFeatureProfile().getInfo(old_id));
    }
    merged_index->GetPqCodeProfile().insert(new_id, ivf_index->GetPqCodeProfile().getInfo(old_id));
    merged_index->GetRankScoreProfile().insert(new_id, ivf_index->GetRankScoreProfile().getInfo(old_id));
    if (merged_index->MultiAgeMode()) {
        merged_index->GetDocCreateTimeProfile().insert(new_id, merged_index->GetDocCreateTimeProfile().getInfo(old_id));
    }

    std::vector<std::vector<SlotIndex>>& slots_profile = indexes_slots_profile_.at(index_offset);
    std::vector<SlotIndex>& slots = slots_profile.at(old_id);
    for (size_t i = 0; i < slots.size(); i++) {
        SlotIndex label = slots.at(i);
        // i is globalId
        bool ret = merged_index->GetCoarseIndex().addDoc(label, new_id);
        if (!ret) {
            LOG_ERROR("insert doc[%u] error with id[%u] from indexes[%lu]", new_id, old_id, index_offset);
            return -1;
        }
    }

    return 0;
}

template <typename T>
int IvfFastScanMerger::DoMerge(const std::vector<Index::Pointer> &indexes,
                            const std::vector<std::pair<size_t, T>> merge_map,
                            bool by_pk) {
    if (!CheckSize(indexes, merge_map.size())) {
        LOG_ERROR("merge size check error.");
        return -1;
    }

    if (indexes.empty()) {
        LOG_ERROR("empty index vector.");
        return -1;
    }

    size_t total_doc_num = 0;
    for (const auto& index : indexes) {
        const IvfFastScanIndex* ivf_index = dynamic_cast<IvfFastScanIndex*>(index.get());
        if (ivf_index == nullptr) {
            LOG_ERROR("convert to group ivf index failed.");
            return -1;
        }

        total_doc_num += ivf_index->GetDocNum();
        if (GenerateSlotsProfile(ivf_index) != 0) {
            LOG_ERROR("generate slots profile failed.");
            return -1;
        }
    }

    auto front_index = dynamic_cast<IvfFastScanIndex*>(indexes.front().get());
    merged_index_.CopyInit(front_index, total_doc_num);

    int doc_num_add_suc = 0;
    
    for (size_t i = 0; i < merge_map.size(); i++) {
        const std::pair<size_t, T>& info_pair = merge_map.at(i);
        if (info_pair.first >= indexes.size()) {
            LOG_ERROR("index offset %lu to be merged exceed index num.", info_pair.first);
            return -1;
        }
        size_t index_offset = info_pair.first;

        if (DoMergeEachIvf(i, info_pair.second, index_offset, indexes, by_pk, &merged_index_) != 0) {
            LOG_WARN("do merge each ivf failed. continue.");
            continue;
        }
        doc_num_add_suc++;
    }
    
    merged_index_.SetDocNum(doc_num_add_suc);

    size_t pq_fragment_num = merged_index_.GetPqCentroidResource().getIntegrateMeta().fragmentNum;
    auto slot_num = merged_index_.GetCentroidResourceManager().GetTotalCentroidsNum();
    std::vector<size_t> codeCounts;
    codeCounts.reserve(slot_num);
    for (size_t s = 0; s < slot_num; s++) {
        CoarseIndex<BigBlock>::PostingIterator posting = merged_index_.GetCoarseIndex().search(s);
        size_t roundUp32DocNum = FastScanIndex::ROUNDUP32(posting.getDocNum());
        size_t codeCount = roundUp32DocNum * pq_fragment_num / 2;
        codeCounts.push_back(codeCount);
    }

    if (!merged_index_.InitFastScanIndex(codeCounts)) {
        LOG_ERROR("failed to init fastscan codes");
        return -1;
    }
    if (!PackFastScanCodes()) {
        LOG_ERROR("failed to pack fastscan codes");
        return -1;
    }
    return 0;
}

bool IvfFastScanMerger::PackFastScanCodes() {
    size_t pq_fragment_num = merged_index_.GetPqCentroidResource().getIntegrateMeta().fragmentNum;
    auto slot_num = merged_index_.GetCentroidResourceManager().GetTotalCentroidsNum();
    for (size_t slot = 0; slot < slot_num; slot++) {
        CoarseIndex<BigBlock>::PostingIterator posting = merged_index_.GetCoarseIndex().search(slot);
        std::vector<uint8_t> oriCode;
        oriCode.reserve(posting.getDocNum() * pq_fragment_num);
        while(!posting.finish()) {
            auto docid = posting.next();
            const uint8_t *pqcode =
                reinterpret_cast<const uint8_t *>(merged_index_.GetPqCodeProfile().getInfo(docid));
            for (size_t j = 0; j < pq_fragment_num; j++) {
                oriCode.push_back(pqcode[j]);
            }
        }
        size_t roundUp32DocNum = FastScanIndex::ROUNDUP32(posting.getDocNum());
        std::vector<uint8_t> packedCode;
        packedCode.resize(roundUp32DocNum * pq_fragment_num / 2);
        memset(packedCode.data(), 0, posting.getDocNum() * pq_fragment_num / 2);

        // permutation for cache-line optimization
        const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

        uint8_t* codes = oriCode.data();

        uint8_t* codes2 = packedCode.data();

        for (size_t i0 = 0; i0 < roundUp32DocNum; i0 += 32) {
            for (size_t sq = 0; sq < pq_fragment_num; sq += 2) {
                std::array<uint8_t, 32> c0, c1;
                FastScanIndex::get_matrix_column(
                        codes, posting.getDocNum(), pq_fragment_num, i0, sq, c0);
                FastScanIndex::get_matrix_column(
                        codes, posting.getDocNum(), pq_fragment_num, i0, sq + 1, c1);
                for (int j = 0; j < 16; j++) {
                    uint8_t d0, d1;
                    d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                    d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                    codes2[j] = d0;
                    codes2[j + 16] = d1;
                }
                codes2 += 32;
            }
        }
        for (size_t k = 0; k < packedCode.size(); k++) {
            merged_index_.GetFastScanIndex().addCode(slot, packedCode[k]);
        }
    }
    return true;
}

int IvfFastScanMerger::GenerateSlotsProfile(const IvfFastScanIndex* ivf_index) {
    std::vector<std::vector<SlotIndex>> slots_profile; //docid->slot indexes. 一个doc会有多个slotindex
    size_t slot_num = ivf_index->GetCoarseIndex().getSlotNum();
    // 用空的slotindex初始化
    slots_profile.resize(ivf_index->GetCoarseIndex().getUsedDocNum(), std::vector<SlotIndex>());
    for (size_t i = 0; i < slot_num; i++) {
        CoarseIndex<BigBlock>::PostingIterator iter = ivf_index->GetCoarseIndex().search(i);
        while (!iter.finish()) {
            docid_t docid = iter.next();
            slots_profile[docid].push_back(i);
        }
    }

    indexes_slots_profile_.push_back(slots_profile);
    return 0;
}

int IvfFastScanMerger::MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) {
    return DoMerge(indexes, id_map, false);
}

int IvfFastScanMerger::MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) {
    return DoMerge(indexes, pk_map, true);
}

//! Dump index into file or memory
int IvfFastScanMerger::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    return -1;
}

const void * IvfFastScanMerger::DumpIndex(size_t *size) {
    const void *data = nullptr;
    if (merged_index_.Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump merged index.");
        *size = 0;
        return nullptr;
    }
    return data;
}

bool IvfFastScanMerger::CheckSize(const std::vector<Index::Pointer> &indexes, size_t new_size) const {
    // check
    size_t total_size = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
        total_size += indexes[i]->GetDocNum();
    }

    //maybe some delete in new so new_size maybe smaller or equal
    if (total_size < new_size) {
        LOG_ERROR("merge size error. ori: %lu, new: %lu", total_size, new_size);
        return false;
    }

    return true;
}

MERCURY_NAMESPACE_END(core);
