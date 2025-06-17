#include <gtest/gtest.h>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "framework/utility/time_helper.h"
#include "common/common_define.h"

#define protected public
#define private public
#include "common/vecs_file_holder.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

class VecsFileHolderTest: public testing::Test
{
public:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};


TEST_F(VecsFileHolderTest, TestIterator)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1.0f);

    vector<float> expectedValues;
    vector<key_t> expectedKeys;
    const size_t recordNum = 10;
    const size_t dim = 256;
    for (size_t i = 0; i < recordNum; ++i) {
        size_t key = i + 1;
        expectedKeys.push_back(key);
        for (size_t k = 0; k < dim; ++k) {
            float v = dist(gen);
            expectedValues.push_back(v);
        }
    }
    ofstream out("out.vecs", std::ios::binary);
    EXPECT_TRUE(out.is_open());
    out.write((char*)expectedValues.data(), expectedValues.size()*sizeof(float));
    out.write((char*)expectedKeys.data(), expectedKeys.size()*sizeof(key_t));
    out.close();

    IndexMeta meta;
    meta.setMeta(IndexMeta::kTypeFloat, 256);
    VecsFileHolder holder(meta);
    EXPECT_TRUE(holder.load("out.vecs"));


    size_t index = 0;
    for (auto iter = holder.createIterator(); iter && iter->isValid(); iter->next())  {
        EXPECT_EQ(expectedKeys[index], iter->key());
        for (size_t k = 0; k < dim; ++k ) {
            ASSERT_FLOAT_EQ(expectedValues[index*dim+k], ((float*)iter->data())[k]);
        }
        index++;
    }

    system("rm -f out.vecs");
}

TEST_F(VecsFileHolderTest, TestIteratorWithCat)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1.0f);

    vector<float> expectedValues;
    vector<key_t> expectedKeys;
    vector<cat_t> expectedCats;
    const size_t recordNum = 10;
    const size_t dim = 256;
    for (size_t i = 0; i < recordNum; ++i) {
        size_t key = i + 1;
        expectedKeys.push_back(key);
        size_t cat = i + 10;
        expectedCats.push_back(cat);
        for (size_t k = 0; k < dim; ++k) {
            float v = dist(gen);
            expectedValues.push_back(v);
        }
    }
    ofstream out("out.vecs", std::ios::binary);
    EXPECT_TRUE(out.is_open());
    out.write((char*)expectedValues.data(), expectedValues.size()*sizeof(float));
    out.write((char*)expectedKeys.data(), expectedKeys.size()*sizeof(key_t));
    out.write((char*)expectedCats.data(), expectedCats.size()*sizeof(cat_t));
    out.close();

    IndexMeta meta;
    meta.setMeta(IndexMeta::kTypeFloat, 256);
    bool catEnabled = true;
    VecsFileHolder holder(meta, catEnabled);
    EXPECT_TRUE(holder.load("out.vecs"));


    size_t index = 0;
    for (auto iter = holder.createIterator(); iter && iter->isValid(); iter->next())  {
        EXPECT_EQ(expectedKeys[index], iter->key());
        EXPECT_EQ(expectedCats[index], iter->cat());
        for (size_t k = 0; k < dim; ++k ) {
            ASSERT_FLOAT_EQ(expectedValues[index*dim+k], ((float*)iter->data())[k]);
        }
        index++;
    }

    system("rm -f out.vecs");
}

}; // namespace mercury
