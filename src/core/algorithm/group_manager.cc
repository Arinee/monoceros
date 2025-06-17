#include <algorithm>
#include "group_manager.h"

MERCURY_NAMESPACE_BEGIN(core);
gindex_t GroupManager::GetGroupIndex(const GroupInfo& group_info) const {
    // binary search
    auto iter =  std::lower_bound(group_vec_.begin(), group_vec_.end(),
                                  group_info, GroupSorter);
    if (iter == group_vec_.end() || !(group_info.id == iter->id && group_info.level == iter->level)) {
        LOG_WARN("group info not found. Level: %d, id: %d", group_info.level, group_info.id);
        return INVALID_GROUP_INDEX;
    }

    return std::distance(group_vec_.begin(), iter);
}

int GroupManager::Create(const std::vector<GroupInfo>&& raw_groups) {
    if (raw_groups.size() <= 0) {
        LOG_ERROR("empty raw_groups");
        return -1;
    }

    group_vec_.assign(raw_groups.begin(), raw_groups.end());
    std::sort(group_vec_.begin(), group_vec_.end(), GroupSorter);

    uint32_t last_level = group_vec_.at(0).level;
    uint32_t last_id = group_vec_.at(0).id;
    for (size_t i = 1; i < group_vec_.size(); i++) {
        uint32_t level = group_vec_.at(i).level;
        uint32_t id = group_vec_.at(i).id;
        if (level == last_level && id == last_id) {
            LOG_ERROR("duplicated group level && id: %u, %u", level, id);
            return -1;
        }

        last_level = level;
        last_id = id;
    }

    if (group_vec_.size() > 1 && group_vec_.at(1).level == 0) {
        LOG_ERROR("group info contain more than 1 zero group.");
        return -1;
    }
    return 0;
}

bool GroupSorter(const GroupInfo& lhs, const GroupInfo& rhs) {
    if (lhs.level != rhs.level) {
        return lhs.level < rhs.level;
    }

    if (lhs.level == 0) {
        return false;
    }

    return lhs.id < rhs.id;
}

int GroupManager::Load(const void* data, size_t size) {
    group_vec_.resize(size / sizeof(GroupInfo));
    if (memcpy(group_vec_.data(), data, size) == nullptr) {
        return -1;
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);
