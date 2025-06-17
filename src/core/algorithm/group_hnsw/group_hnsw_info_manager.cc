#include "group_hnsw_info_manager.h"

MERCURY_NAMESPACE_BEGIN(core);

int GroupHnswManager::Create(const std::vector<std::pair<GroupInfo, GroupHnswInfo>>&& group_hnsw_infos) {
    if (group_hnsw_infos.size() <= 0) {
        LOG_ERROR("empty raw_groups");
        return -1;
    }
    group_hnsw_infos_.assign(group_hnsw_infos.begin(), group_hnsw_infos.end());
    std::sort(group_hnsw_infos_.begin(), group_hnsw_infos_.end(), GroupHnswSorter);
    return 0;
}

void GroupHnswManager::CalGroupOffset() {
    LOG_INFO("-------CalGroupOffset begin-------");
    uint64_t group_offset = sizeof(CoarseHnswIndex::Header);
    // uint64_t group_offset = 0;
    for (auto &group_hnsw_info : group_hnsw_infos_) {
        group_hnsw_info.second.offset = group_offset;
        // LOG_INFO("group level:%u id: %u group_offset: %lu", group_hnsw_info.first.level, group_hnsw_info.first.id, group_offset);
        if(group_hnsw_info.second.is_build_hnsw == false) {
            group_offset = group_offset + sizeof(CoarseHnswIndex::GroupHeader) + group_hnsw_info.second.doc_total_num * sizeof(docid_t);
        } else {
            uint64_t offset_size = group_hnsw_info.second.doc_total_num * sizeof(uint64_t);
            uint64_t base_neighbor_total = base_neighbor_cnt_ * group_hnsw_info.second.doc_total_num;
            uint64_t upper_doc_total = 0;
            float low_level_doc_cnt = static_cast<float>(group_hnsw_info.second.doc_total_num);
            for (uint32_t i = 0; i < max_level_; ++i) {
                low_level_doc_cnt /= scalingFactor_;
                upper_doc_total += std::ceil(low_level_doc_cnt);
            }
            uint64_t upper_neighbor_total = upper_neighbor_cnt_ * upper_doc_total;
            // 邻居信息总大小，其中包含每个doc在每层所有邻居元信息，以及邻居docid信息
            uint64_t neighbor_size = (group_hnsw_info.second.doc_total_num + upper_doc_total) * sizeof(CoarseHnswIndex::NeighborListHeader) + \
                                    (base_neighbor_total + upper_neighbor_total) * sizeof(docid_t);
            // 防止实际数量大于概率计算，空间膨胀 1.1 倍
            neighbor_size *= 1.1;
            // LOG_INFO("neighbor_size: %lu", neighbor_size);
            // 每个doc邻居信息之前添加一个docid_t记录该doc的全局docid
            uint64_t global_doc_id_size = group_hnsw_info.second.doc_total_num * sizeof(docid_t);
            group_offset = group_offset + sizeof(CoarseHnswIndex::GroupHeader) + offset_size + neighbor_size \
                            + global_doc_id_size;
        }
        group_hnsw_info.second.end_offset = group_offset;
        // LOG_INFO("group level:%u id: %u group_end_offset: %lu", group_hnsw_info.first.level, group_hnsw_info.first.id, group_offset);
        group_hnsw_infos_map_.insert(group_hnsw_info);
    }
    //存储内存总长度
    capacity_ = group_offset;
    LOG_INFO("coarse_hnsw_index_ capacity_: %lu", capacity_);
    LOG_INFO("-------CalGroupOffset end-------");
}

int GroupHnswManager::Load(const void* data, size_t size) {
    group_hnsw_infos_.resize(size / sizeof(std::pair<GroupInfo, GroupHnswInfo>));
    if (memcpy(reinterpret_cast<void *>(group_hnsw_infos_.data()), data, size) == nullptr) {
        return -1;
    }
    for (auto &group_hnsw_info : group_hnsw_infos_) {
        group_hnsw_infos_map_.insert(group_hnsw_info);
    }
    LOG_INFO("group_hnsw_infos_.size(): %zd", group_hnsw_infos_.size());
    return 0;
}

bool GroupHnswSorter(const std::pair<GroupInfo, GroupHnswInfo>& lhs, const std::pair<GroupInfo, GroupHnswInfo>& rhs) {
    // 不同level升序排列
    if (lhs.first.level != rhs.first.level) {
        return lhs.first.level < rhs.first.level;
    }
    if (lhs.first.level == 0) {
        return false;
    }
    // 相同level按id升序排列
    return lhs.first.id < rhs.first.id;
}

MERCURY_NAMESPACE_END(core);
