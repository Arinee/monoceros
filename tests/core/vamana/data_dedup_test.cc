/// Copyright (c) 2023, xiaohongshu Inc. All rights reserved.
/// Author: shiyang <shiyang1@xiaohongshu.com>
/// Created: 2023-11-04 11:00

#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <numeric>
#include <omp.h>

#define protected public
#define private public
#include "src/core/algorithm/algorithm_factory.h"
#undef protected
#undef private

MERCURY_NAMESPACE_BEGIN(core);

class DataDedupTest: public testing::Test
{
public:
    void SetUp()
    {
        // input data:
        input_filename_ = "tests/core/vamana/test_data_shoe_5m/img_shoe_5m.full";
        // output data:
        out_filename_ = "tests/core/vamana/test_data_shoe_5m/img_shoe_5m.dedup";
    }

    void TearDown()
    {

    }

    std::string input_filename_, out_filename_;
};

std::string trimWhitespace(const std::string& str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::map<std::string, std::string> processLine(const std::string& line) {
    std::map<std::string, std::string> record;
    std::istringstream iss(line);
    std::string field;
    const char delimiter = '\x1f';

    while (std::getline(iss, field, delimiter)) {
        size_t pos = field.find('=');
        if (pos != std::string::npos) {
            std::string key = trimWhitespace(field.substr(0, pos));
            std::string value = trimWhitespace(field.substr(pos + 1));
            record[key] = value;
        }
    }

    return record;
}

TEST_F(DataDedupTest, Dedup) {

    GTEST_SKIP() << "Skipping DataDedupTest.Dedup";

    std::ofstream out_file(out_filename_);

    std::ifstream in_file(input_filename_);
    if (!in_file.is_open()) {
        LOG_ERROR("Error: Could not open base_file with path: %s", input_filename_.c_str());
        return;
    }
    std::string lines;
    int index = 0;
    int dupCount = 0;
    std::unordered_set<std::string> deDup;
    deDup.reserve(1000000);
    while (std::getline(in_file, lines, '\x1e')) {
        std::map<std::string, std::string> record = processLine(lines);

        auto id_itr = record.find("id");
        if (id_itr != record.end()) {
            if (deDup.find(id_itr->second) == deDup.end()) {
                deDup.insert(id_itr->second);
                out_file << lines << '\x1e';
            } else {
                dupCount++;
            }
            index++;
        }
    }
    out_file.close();
}

MERCURY_NAMESPACE_END(core);
