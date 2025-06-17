#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <stdexcept>

class QueryFileReader {
public:
    QueryFileReader(const std::string& filename);
    std::vector<std::pair<std::string, std::string>> getRequests();

private:
    std::string filename_;
};
