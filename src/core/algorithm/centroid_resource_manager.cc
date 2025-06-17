#include <sstream>
#include "centroid_resource_manager.h"
#include "src/core/framework/index_logger.h"

MERCURY_NAMESPACE_BEGIN(core);

CentroidResource& CentroidResourceManager::GetCentroidResource(gindex_t group_index) {
    if (group_index >= centroid_resources_.size()) {
        LOG_ERROR("group_index %u is out of bound %lu", group_index, centroid_resources_.size());
    }
    return centroid_resources_.at(group_index);
}

const CentroidResource& CentroidResourceManager::GetCentroidResource(gindex_t group_index) const {
    if (group_index >= centroid_resources_.size()) {
        LOG_ERROR("group_index %u is out of bound %lu", group_index, centroid_resources_.size());
    }
    return centroid_resources_.at(group_index);
}

// 从rough_base load，该rough_base包含多个centroid_resource对应的rough_matrix
int CentroidResourceManager::LoadRough(const void *rough_base, size_t rough_len) {
    size_t offset = 0;
    char * buf = (char *)rough_base;
    while (rough_len > offset) {
        uint32_t part_len = *((uint32_t *)(buf + offset));
        offset += sizeof(uint32_t);
        CentroidResource cr;
        cr.init(buf + offset, part_len);
        AddCentroidResource(std::move(cr));
        offset += part_len;
    }

    return 0;
}

int CentroidResourceManager::DumpRoughMatrix(std::string &rough_string) const {
    if (centroid_resources_.size() <= 0) {
        LOG_ERROR("no centroid_resources, nothing to dump.");
        return -1;
    }

    std::vector<std::string> parts;
    size_t total_size = 0;
    for (size_t i = 0; i < centroid_resources_.size(); i++) {
        std::string part;
        centroid_resources_.at(i).dumpRoughMatrix(part);
        total_size += part.size();
        parts.push_back(part);
    }

    rough_string.reserve(total_size + sizeof(uint32_t) * parts.size());
    for (size_t i = 0; i < parts.size(); i++) {
        uint32_t size = parts.at(i).size();
        rough_string.append((char*)&(size), sizeof(uint32_t));
        rough_string.append(parts.at(i));
    }
    return 0;
}

int CentroidResourceManager::AddCentroidResource(const CentroidResource&& centroid_resource) {
    off_t start_offset = 0;
    if (group_starts_.size() == 0) {
        group_starts_.push_back(0);
    }
    start_offset += group_starts_.at(group_starts_.size() - 1)
        + centroid_resource.getLeafCentroidNum();
    group_starts_.push_back(start_offset);
    centroid_resources_.push_back(std::move(centroid_resource));
    return 0;
}

//根据group偏移，计算在coarseindex的slotindex中的位置
off_t CentroidResourceManager::GetSlotIndex(gindex_t group_index, uint32_t label) const {
    if (group_index >= group_starts_.size()) {
        LOG_ERROR("Invalid group_index: %u, max is %lu", group_index, group_starts_.size()-1);
        return INVALID_SLOT_INDEX;
    }
    return group_starts_.at(group_index) + label;
}

gindex_t CentroidResourceManager::GetGroupIndex(uint32_t slot_index) const {
    for (size_t i = group_starts_.size() - 1; i >=0; i--) {
        if (group_starts_.at(i) <= slot_index) {
            return i;
        }
    }

    return INVALID_GROUP_INDEX;
}

size_t CentroidResourceManager::GetTotalCentroidsNum() const {
    size_t total_num = 0;
    for (size_t i = 0; i < centroid_resources_.size(); i++) {
        total_num += centroid_resources_.at(i).getLeafCentroidNum();
    }

    return total_num;
};

size_t CentroidResourceManager::GetCentroidsNum(gindex_t group_index) const {
    return centroid_resources_.at(group_index).getLeafCentroidNum();
};

MERCURY_NAMESPACE_END(end);
