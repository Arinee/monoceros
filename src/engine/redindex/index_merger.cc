/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-13 18:08

#include "index_merger.h"

namespace mercury {
namespace redindex {

int IndexMerger::PreUpdate(const std::vector<RedIndex::Pointer>& indexes,
                              const std::vector<RedIndexDocid>& new_redindex_docids) {
    // check
    size_t total_size = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
        total_size += indexes[i]->get_current_doc_num();
    }
    if (total_size != new_redindex_docids.size()) {
        fprintf(stderr, "redindex docid size error, expected: %lu, actual: %lu\n",
                total_size, new_redindex_docids.size());
        return -1;
    }
    new_redindex_docids_.assign(new_redindex_docids.begin(), new_redindex_docids.end());
    return 0;
}

size_t GetIndexOffset(RedIndexDocid* old_redindex_docid, const std::vector<RedIndex::Pointer>& indexes) {
    size_t offset = *old_redindex_docid;
    size_t start = 0;
    for (size_t i = 0; i < indexes.size(); i++) {
        if (offset >= start && offset < start + indexes.at(i)->get_current_doc_num()) {
            return i;
        }

        *old_redindex_docid -= indexes.at(i)->get_current_doc_num();
        start += indexes.at(i)->get_current_doc_num();
    }

    return mercury::core::INVALID_DOC_ID;
}

bool ConvertCoreMap(mercury::core::MergeIdMap& id_map, const std::vector<RedIndexDocid>& new_redindex_docids,
                    const std::vector<RedIndex::Pointer>& indexes) {
    id_map.resize(new_redindex_docids.size());
    size_t delete_num = 0;
    for (size_t i = 0; i < new_redindex_docids.size(); i++) {
        if (new_redindex_docids.at(i) != -1) {
            RedIndexDocid old_id = i;
            size_t offset = GetIndexOffset(&old_id, indexes);
            if (offset == mercury::core::INVALID_DOC_ID) {
                std::cerr << "GetIndexOffset failed." << std::endl;
                return false;
            }

            id_map[new_redindex_docids.at(i)] = std::pair<size_t, mercury::core::docid_t>(offset, old_id);
        } else {
            delete_num++;
        }
    }

    std::cout << "delete_num: " << delete_num << std::endl;
    id_map.resize(new_redindex_docids.size() - delete_num);
    return true;
}

void ConvertCoreIndex(std::vector<mercury::core::Index::Pointer>& core_indexes,
                      const std::vector<RedIndex::Pointer>& indexes) {
    for (size_t i = 0; i < indexes.size(); i++) {
        core_indexes.push_back(indexes.at(i)->GetCoreIndex());
    }
}

// Make sure _indexes[j]->GetNewRedIndexDocids() is set
int IndexMerger::Merge(const std::vector<RedIndex::Pointer>& indexes) {
    //convert to core implement
    if (indexes.empty()) {
        std::cerr << "empty index vector." << std::endl;
        return -1;
    }

    mercury::core::MergeIdMap id_map;
    if (ConvertCoreMap(id_map, new_redindex_docids_, indexes) == false) {
        std::cerr << "convert core map failed." << std::endl;
        return -1;
    }

    std::vector<mercury::core::Index::Pointer> core_indexes;
    ConvertCoreIndex(core_indexes, indexes);

    int ret = core_merger_->MergeByDocid(core_indexes, id_map);
    if (ret != 0) {
        std::cerr << "do mergebydocid failed." << std::endl;
        return -1;
    }

    return 0;
}

const void * IndexMerger::Dump(size_t *size) {
    return core_merger_->DumpIndex(size);
}

const void * IndexMerger::DumpCentroid(size_t *size) {
    return core_merger_->DumpCentroid(size);
}

size_t IndexMerger::UsedMemoryInMerge() {
    //TODO why
    return 0xFFFF;
}

} // namespace redindex
} // namespace mercury
