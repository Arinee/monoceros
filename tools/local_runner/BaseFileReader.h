#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "src/core/common/common_define.h"

class BaseFileReader {
public:
    std::map<int, std::pair<std::string, std::string>> readFileAndProcessLines(const std::string& fileName);

private:
    std::string trimWhitespace(const std::string& str);
    std::map<std::string, std::string> processLine(const std::string& line);
};
