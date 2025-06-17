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
#include "common/txt_file_holder.h"
#undef protected
#undef private

using namespace std;

namespace mercury {

class TxtFileHolderTest: public testing::Test
{
public:
    void SetUp() override
    {
    }

    void TearDown() override
    {
    }
};


TEST_F(TxtFileHolderTest, TestIterator)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1.0f);

    vector<float> expectedValues;
    vector<key_t> expectedKeys;
    vector<string> records;
    const size_t recordNum = 10;
    const size_t dim = 256;
    for (size_t i = 0; i < recordNum; ++i) {
        string record;
        size_t key = i + 1;
        expectedKeys.push_back(key);
        record += to_string(key) + "; ";
        for (size_t k = 0; k < dim; ++k) {
            float v = dist(gen);
            expectedValues.push_back(strtof(to_string(v).c_str(), nullptr));
            record += to_string(v);
            if (k != dim - 1) {
                record += ", ";
            }
        }
        records.push_back(record);
    }
    ofstream out("out.txt");
    EXPECT_TRUE(out.is_open());
    for (size_t i = 0; i < records.size(); ++i) {
        out << records[i] << endl;
    }
    out.close();

    TxtFileHolder txtHolder(IndexMeta::kTypeFloat, 256, ";", ",");
    EXPECT_TRUE(txtHolder.load("out.txt"));


    size_t index = 0;
    for (auto iter = txtHolder.createIterator(); iter && iter->isValid(); iter->next())  {
        EXPECT_EQ(expectedKeys[index], iter->key());
        for (size_t k = 0; k < dim; ++k ) {
            ASSERT_FLOAT_EQ(expectedValues[index*dim+k], ((float*)iter->data())[k]);
        }
        index++;
    }

    system("rm -f out.txt");
}


TEST_F(TxtFileHolderTest, TestCatIterator)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1.0f);

    vector<key_t> expectedCats;
    vector<key_t> expectedKeys;
    vector<float> expectedValues;
    vector<string> records;
    const size_t recordNum = 10;
    const size_t dim = 256;
    for (size_t i = 0; i < recordNum; ++i) {
        string record;
        size_t cat = i + 10;
        expectedCats.push_back(cat);
        record += to_string(cat) + "; ";
        size_t key = i + 1;
        expectedKeys.push_back(key);
        record += to_string(key) + "; ";
        for (size_t k = 0; k < dim; ++k) {
            float v = dist(gen);
            record += to_string(v);
            expectedValues.push_back(atof(to_string(v).c_str()));
            if (k != dim - 1) {
                record += ", ";
            }
        }
        records.push_back(record);
    }
    ofstream out("out.txt");
    EXPECT_TRUE(out.is_open());
    for (size_t i = 0; i < records.size(); ++i) {
        out << records[i] << endl;
    }
    out.close();

    TxtFileHolder txtHolder(IndexMeta::kTypeFloat, 256, ";", ",", true);
    EXPECT_TRUE(txtHolder.load("out.txt"));


    size_t index = 0;
    for (auto iter = txtHolder.createIterator(); iter && iter->isValid(); iter->next())  {
        EXPECT_EQ(expectedCats[index], iter->cat());
        EXPECT_EQ(expectedKeys[index], iter->key());
        for (size_t k = 0; k < dim; ++k ) {
            ASSERT_FLOAT_EQ(expectedValues[index*dim+k], ((float*)iter->data())[k]);
        }
        index++;
    }

    system("rm -f out.txt");
}

TEST_F(TxtFileHolderTest, IgnoreInvalidLines_Iterator)
{
    ofstream out("out.txt");
    EXPECT_TRUE(out.is_open());
    out << "" << std::endl;
    out << "1" << std::endl;
    out << "1;" << std::endl;
    out << "1;1.0" << std::endl;
    out.close();

    TxtFileHolder txtHolder(IndexMeta::kTypeFloat, 256, ";", ",");
    EXPECT_TRUE(txtHolder.load("out.txt"));

    size_t index = 0;
    for (auto iter = txtHolder.createIterator(); iter && iter->isValid(); iter->next())  {
        ++index;
    }
    EXPECT_EQ(index, (size_t)0);

    system("rm -f out.txt");
}


TEST_F(TxtFileHolderTest, IgnoreInvalidLines_CatIterator)
{
    ofstream out("out.txt");
    EXPECT_TRUE(out.is_open());
    out << "" << std::endl;
    out << "1" << std::endl;
    out << "1;" << std::endl;
    out << "1;2" << std::endl;
    out << "1;2;" << std::endl;
    out << "1;2;1.0" << std::endl;
    out.close();

    TxtFileHolder txtHolder(IndexMeta::kTypeFloat, 256, ";", ",", true);
    EXPECT_TRUE(txtHolder.load("out.txt"));

    size_t index = 0;
    for (auto iter = txtHolder.createIterator(); iter && iter->isValid(); iter->next())  {
        ++index;
    }
    EXPECT_EQ(index, (size_t)0);

    system("rm -f out.txt");
}

}; // namespace mercury
