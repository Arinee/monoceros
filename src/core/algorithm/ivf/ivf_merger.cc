#include "ivf_merger.h"

MERCURY_NAMESPACE_BEGIN(core);

//! Initialize Merger
int IvfMerger::Init(const IndexParams &params) {
    return 0;
}

template <typename T>
int DoMergeEachIvf(docid_t new_id, T old_id_pk, size_t index_offset,
                              const std::vector<Index::Pointer> &indexes, bool by_pk,
                              IvfIndex* merged_index) {
    if (merged_index == nullptr) {
        LOG_ERROR("merged_index is null");
        return -1;
    }

    pk_t pk = old_id_pk;
    docid_t old_id = old_id_pk;

    assert(index_offset != INVALID_DOC_ID);
    auto ivf_index = dynamic_cast<IvfIndex*>(indexes.at(index_offset).get());
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

    SlotIndex* label = (SlotIndex*)ivf_index->GetSlotIndexProfile().getInfo(old_id);
    // i is globalId
    bool ret = merged_index->GetCoarseIndex().addDoc(*label, new_id);
    if (!ret) {
        LOG_ERROR("insert doc[%u] error with id[%u] from indexes[%lu]", new_id, old_id, index_offset);
        return -1;
    }

    ret = merged_index->GetSlotIndexProfile().insert(new_id, label);
    if (!ret) {
        LOG_ERROR("insert doc[%u] error with new_docid[%u] from indexes[%lu]",
                  new_id, new_id, index_offset);
        return -1;
    }

    return 0;
}

template <typename T>
int IvfMerger::DoMerge(const std::vector<Index::Pointer> &indexes,
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

    // Init resource get from first index
    size_t total_doc_num = 0;
    for (const auto& index : indexes) {
        total_doc_num += index->GetDocNum();
    }

    auto front_index = dynamic_cast<IvfIndex*>(indexes.front().get());
    merged_index_.CopyInit(front_index, total_doc_num);

    for (size_t i = 0; i < merge_map.size(); i++) {
        const std::pair<size_t, T>& info_pair = merge_map.at(i);
        size_t index_offset = info_pair.first;

        if (DoMergeEachIvf(i, info_pair.second, index_offset, indexes, by_pk, &merged_index_) != 0) {
            LOG_WARN("do merge each ivf failed. continue.");
            continue;
        }
    }

    return 0;
}

int IvfMerger::MergeByDocid(const std::vector<Index::Pointer> &indexes, const MergeIdMap& id_map) {
    return DoMerge(indexes, id_map, false);
}

int IvfMerger::MergeByPk(const std::vector<Index::Pointer> &indexes, const MergePkMap& pk_map) {
    return DoMerge(indexes, pk_map, true);
}

//! Dump index into file or memory
int IvfMerger::DumpIndex(const std::string &path, const IndexStorage::Pointer &stg) {
    return -1;
}

const void * IvfMerger::DumpIndex(size_t *size) {
    const void *data = nullptr;
    if (merged_index_.Dump(data, *size) != 0) {
        LOG_ERROR("Failed to dump merged index.");
        *size = 0;
        return nullptr;
    }
    return data;
}

bool IvfMerger::CheckSize(const std::vector<Index::Pointer> &indexes, size_t new_size) const {
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
