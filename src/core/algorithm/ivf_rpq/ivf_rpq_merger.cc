#include "ivf_rpq_merger.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Merger
int IvfRpqMerger::Init(const IndexParams &params) {
    merged_index_.SetIndexParams(params);
    return 0;
}

template <typename T>
int IvfRpqMerger::DoMergeEachIvf(docid_t new_id, T old_id_pk, size_t index_offset,
                              const std::vector<Index::Pointer> &indexes, bool by_pk,
                              IvfRpqIndex* merged_index) {
    if (merged_index == nullptr) {
        LOG_ERROR("merged_index is null");
        return -1;
    }

    pk_t pk = old_id_pk;
    docid_t old_id = old_id_pk;

    assert(index_offset != INVALID_DOC_ID);
    auto ivf_index = dynamic_cast<IvfRpqIndex*>(indexes.at(index_offset).get());
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
        if (merged_index->EnableQuantize()) {
            float vmin = merged_index_.GetVmin();
            float vmax = merged_index_.GetVmax();
            size_t d = merged_index_.GetIndexMeta().dimension();
            const float* original_vector = static_cast<const float*>(ivf_index->GetFeatureProfile().getInfo(old_id));
            std::vector<uint8_t> encoded_vector(d);
            encode_vector(original_vector, encoded_vector.data(), vmin, vmax, d);

            std::vector<float> decoded_vector(d);
            decode_vector(encoded_vector.data(), decoded_vector.data(), vmin, vmax, d);

            merged_index->GetFeatureProfile().insert(new_id, encoded_vector.data());
        } else {
            merged_index->GetFeatureProfile().insert(new_id, ivf_index->GetFeatureProfile().getInfo(old_id));
        }
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

    //TODO, handle doc_num++

    return 0;
}

template <typename T>
int IvfRpqMerger::DoMerge(const std::vector<Index::Pointer> &indexes,
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
    float vmin = HUGE_VAL;
    float vmax = -HUGE_VAL;
    for (const auto& index : indexes) {
        const IvfRpqIndex* ivf_index = dynamic_cast<IvfRpqIndex*>(index.get());
        if (ivf_index->EnableQuantize()) {
            vmin = (vmin < ivf_index->GetVmin()) ? vmin : ivf_index->GetVmin();
            vmax = (vmax > ivf_index->GetVmax()) ? vmax : ivf_index->GetVmax();
        }
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

    auto front_index = dynamic_cast<IvfRpqIndex*>(indexes.front().get());
    merged_index_.CopyInit(front_index, total_doc_num);

    if (merged_index_.EnableQuantize()) {
        LOG_INFO("Vmin = %f and Vmax = %f", vmin, vmax);
        merged_index_.SetVmax(vmax);
        merged_index_.SetVmin(vmin);
    }

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

    return 0;
}

int IvfRpqMerger::GenerateSlotsProfile(const IvfRpqIndex* ivf_index) {
    std::vector<std::vector<SlotIndex>> slots_profile; //docid->slot indexes. 一个doc会有多个slotindex
    size_t slot_num = ivf_index->GetCoarseIndex().getSlotNum();
    // 用空的slotindex初始化
    slots_profile.resize(ivf_index->GetCoarseIndex().getUsedDocNum(), std::vector<SlotIndex>());
    for (size_t i = 0; i < slot_num; i++) {
        CoarseIndex<SmallBlock>::PostingIterator iter = ivf_index->GetCoarseIndex().search(i);
        while (!iter.finish()) {
            docid_t docid = iter.next();
            slots_profile[docid].push_back(i);
        }
    }

    indexes_slots_profile_.push_back(slots_profile);
    return 0;
}

int IvfRpqMerger::MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) {
    return DoMerge(indexes, id_map, false);
}

int IvfRpqMerger::MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) {
    return DoMerge(indexes, pk_map, true);
}

//! Dump index into file or memory
int IvfRpqMerger::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    return -1;
}

const void * IvfRpqMerger::DumpIndex(size_t *size) {
    const void *data = nullptr;
    if (merged_index_.Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump merged index.");
        *size = 0;
        return nullptr;
    }
    return data;
}

bool IvfRpqMerger::CheckSize(const std::vector<Index::Pointer> &indexes, size_t new_size) const {
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
