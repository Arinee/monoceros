/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-12-17 00:59

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#define protected public
#define private public
#include "src/core/utils/note_util.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class NoteUtilTest: public testing::Test
{
public:
    void SetUp()
    {
    }

    void TearDown()
    {
    }

};

TEST_F(NoteUtilTest, TestNoteIdToCreateTimeS) {
    uint32_t create_time_s = 0;

    // test note id length != 24
    bool ret = NoteUtil::NoteIdToCreateTimeS("abcdefghijklmnopqrstuvwxyz", create_time_s);
    ASSERT_FALSE(ret);

    // test hexS string to uint32 failed
    ret = NoteUtil::NoteIdToCreateTimeS("abcdefghijklmnopqrstuvwx", create_time_s);
    ASSERT_FALSE(ret);

    // test valid note id
    ret = NoteUtil::NoteIdToCreateTimeS("12345678ijklmnopqrstuvwx", create_time_s);
    ASSERT_TRUE(ret);
    ASSERT_EQ(create_time_s, 305419896);

    // test real note id
    ret = NoteUtil::NoteIdToCreateTimeS("6489c4bf0000000027028f48", create_time_s);
    ASSERT_TRUE(ret);
    ASSERT_EQ(create_time_s, 1686750399);
}

MERCURY_NAMESPACE_END(core);
