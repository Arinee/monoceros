#ifndef __MERCURY_CORE_GROUP_HNSW_INFO_MANAGER_H__
#define __MERCURY_CORE_GROUP_HNSW_INFO_MANAGER_H__

#include <unordered_map>
#include <utility>
#include "src/core/common/common.h"
#include "src/core/algorithm/group_manager.h"
#include "src/core/framework/index_logger.h"
#include "coarse_hnsw_index.h"


MERCURY_NAMESPACE_BEGIN(core);

struct GroupHnswHash
{
    size_t operator()(const GroupInfo& rhs) const{
        if (rhs.level > 0) {
            return std::hash<level_t>()(rhs.level) ^ std::hash<group_t>()(rhs.id);
        }
        // 0级group忽略ID
        return std::hash<level_t>()(rhs.level);
    }
};
struct GroupHnswCmp
{
    bool operator()(const GroupInfo& lhs, const GroupInfo& rhs) const{
        if (lhs.level == rhs.level && lhs.level == 0) {
            return true;
        }
        return lhs.level == rhs.level && lhs.id == rhs.id;
    }
};

struct GroupHnswInfo {
    uint64_t offset;
    uint64_t end_offset;
    uint32_t doc_total_num;
    bool is_build_hnsw;

    GroupHnswInfo() {};
    GroupHnswInfo(uint32_t doc_total_num, bool is_build_hnsw)
        : offset(0), doc_total_num(doc_total_num), is_build_hnsw(is_build_hnsw) {
    }
};

bool GroupHnswSorter(const std::pair<GroupInfo, GroupHnswInfo>& lhs, const std::pair<GroupInfo, GroupHnswInfo>& rhs);

class GroupHnswManager {
public:
    GroupHnswManager() {}
    ~GroupHnswManager() {}

    int Create(const std::vector<std::pair<GroupInfo, GroupHnswInfo>>&& group_hnsw_infos);
    void CalGroupOffset();
    int Load(const void* data, size_t size);
    inline void SetNeighborCnt(uint64_t upper_neighbor_cnt) {
        upper_neighbor_cnt_ = upper_neighbor_cnt;
        base_neighbor_cnt_ = upper_neighbor_cnt * 2;
    }
    inline void SetMaxLevel(uint32_t max_level) {
        max_level_ = max_level;
    }
    inline void SetScalingFactor(uint64_t scalingFactor) {
        scalingFactor_ = scalingFactor;
    }
    inline uint64_t GetCoarseHnswIndexCapacity() const {
        return capacity_;
    }
    inline uint32_t GetGroupNum() const {
        return group_hnsw_infos_.size();
    }
    inline const GroupHnswInfo& GetGroupHnswInfo(size_t pos) const {
        return group_hnsw_infos_.at(pos).second;
    }
    inline uint64_t GetGroupOffset(const GroupInfo& group_info) const {
        if (group_hnsw_infos_map_.find(group_info) != group_hnsw_infos_map_.end()) {
            return group_hnsw_infos_map_.at(group_info).offset;
        } else {
            return 0;
        }
    }
    inline uint64_t& GetGroupOffsetById(const uint32_t pos) {
        return group_hnsw_infos_.at(pos).second.offset;
    }
    inline bool IsHnswGroup(const GroupInfo& group_info) const {
        if (group_hnsw_infos_map_.find(group_info) != group_hnsw_infos_map_.end()) {
            return group_hnsw_infos_map_.at(group_info).is_build_hnsw;
        } else {
            LOG_ERROR("not found group_level: %u, id: %u in group_hnsw_infos_map_", group_info.level, group_info.id);
            return false;
        }
    }
    inline uint64_t GetCapacity() const {
        return group_hnsw_infos_.size() * sizeof(std::pair<GroupInfo, GroupHnswInfo>);
    }
    inline const void* GetBaseStart() const {
        return group_hnsw_infos_.data();
    }
    void GetLevel3GroupsOffset(std::vector<uint64_t> &offsets) {
        for (auto &group_hnsw_info : group_hnsw_infos_) {
            if (group_hnsw_info.first.level == 3 && group_hnsw_info.second.is_build_hnsw == false) {
                offsets.push_back(group_hnsw_info.second.offset);
            }
        }
    }

private:
    // offset in coarse_hnsw_index of each group
    std::unordered_map<GroupInfo, GroupHnswInfo, GroupHnswHash, GroupHnswCmp> group_hnsw_infos_map_;
    std::vector<std::pair<GroupInfo, GroupHnswInfo>> group_hnsw_infos_;
    uint64_t upper_neighbor_cnt_;       //除第0层每层邻居数
    uint64_t base_neighbor_cnt_;        //第0层邻居数
    uint32_t max_level_;                //HNSW最大层数
    uint64_t scalingFactor_;            //缩放系数
    uint64_t capacity_;                 //coarse_hnsw_index总大小
};

MERCURY_NAMESPACE_END(core);

#endif //__MERCURY_CORE_GROUP_HNSW_INFO_MANAGER_H__