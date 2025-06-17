/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-09-23 10:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

#define protected public
#define private public
#include "src/core/algorithm/group_manager.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class GroupManagerTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(GroupManagerTest, TestSimple)
{
    std::vector<GroupInfo> raw_groups;
    raw_groups.emplace_back(1, 20);
    raw_groups.emplace_back(2, 100);
    raw_groups.emplace_back(0, 0);
    GroupManager group_manager;
    group_manager.Create(std::move(raw_groups));
    ASSERT_EQ(3, group_manager.GetGroupNum());
    ASSERT_EQ(1, group_manager.GetGroupIndex(GroupInfo(1, 20)));
    ASSERT_EQ(INVALID_GROUP_INDEX, group_manager.GetGroupIndex(GroupInfo(1, 21)));
    ASSERT_EQ(0, group_manager.GetGroupInfo(0).level);
}

TEST_F(GroupManagerTest, TestDuplicated)
{
    std::vector<GroupInfo> raw_groups;
    raw_groups.emplace_back(1, 0);
    raw_groups.emplace_back(1, 1);
    raw_groups.emplace_back(0, 1);
    GroupManager group_manager;
    int ret = group_manager.Create(std::move(raw_groups));
    ASSERT_EQ(0, ret);
}

TEST_F(GroupManagerTest, TestDumpLoad)
{
    std::vector<GroupInfo> raw_groups;
    raw_groups.emplace_back(1, 20);
    raw_groups.emplace_back(2, 100);
    raw_groups.emplace_back(0, 0);
    GroupManager group_manager;
    group_manager.Create(std::move(raw_groups));
    const void* data = group_manager.GetBaseStart();
    size_t size = group_manager.GetCapacity();

    GroupManager loaded;
    loaded.Load(data, size);
    ASSERT_EQ(3, loaded.GetGroupNum());
    ASSERT_EQ(1, loaded.GetGroupIndex(GroupInfo(1, 20)));
    ASSERT_EQ(0, loaded.GetGroupInfo(0).level);
}

MERCURY_NAMESPACE_END(core);
