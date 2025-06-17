/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Use: 管理group映射信息，如果包含0级类目，离线数据中应该有，本类不做处理
/// Created: 2019-12-16 10:59

#pragma once

#include <memory>
#include <functional>
#include "src/core/common/common.h"
#include "src/core/framework/index_logger.h"

MERCURY_NAMESPACE_BEGIN(core);

struct GroupInfo {
    level_t level;
    group_t id;
    GroupInfo() {};
    GroupInfo(level_t l, group_t i)
        : level(l), id(i) {};
};

struct GroupHash
{
    size_t operator()(const GroupInfo& rhs) const{
        if (rhs.level > 0) {
            return std::hash<level_t>()(rhs.level) ^ std::hash<group_t>()(rhs.id);
        }

        // 0级group忽略ID
        return std::hash<level_t>()(rhs.level);
    }
};
struct GroupCmp
{
    bool operator()(const GroupInfo& lhs, const GroupInfo& rhs) const{
        if (lhs.level == rhs.level && lhs.level == 0) {
            return true;
        }

        return lhs.level == rhs.level && lhs.id == rhs.id;
    }
};

bool GroupSorter(const GroupInfo& lhs, const GroupInfo& rhs);

class GroupManager {
public:
    GroupManager() {}
    ~GroupManager() {}
    gindex_t GetGroupIndex(const GroupInfo& group_info) const;

    const GroupInfo& GetGroupInfo(gindex_t group_index) const {
        return group_vec_.at(group_index);
    };

    int Load(const void* data, size_t size);
    //int Dump(const void*& data, size_t& size);
    //raw_groups是从训练产出的meta信息中读取出来的
    int Create(const std::vector<GroupInfo>&& raw_groups);

    bool HasZeroGroup() const {
        if (group_vec_.size() == 0) {
            return false;
        }

        if (group_vec_.at(0).level == 0) {
            return true;
        }

        return false;
    }

    const void* GetBaseStart() const {
        return group_vec_.data();
    }

    size_t GetCapacity() const {
        return group_vec_.size() * sizeof(GroupInfo);
    }

    gindex_t GetGroupNum() const {
        return group_vec_.size();
    }

private:
    std::vector<GroupInfo> group_vec_;
};

MERCURY_NAMESPACE_END(core);
