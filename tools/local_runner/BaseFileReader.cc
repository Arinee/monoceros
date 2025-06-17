#include "BaseFileReader.h"

std::string BaseFileReader::trimWhitespace(const std::string& str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::map<std::string, std::string> BaseFileReader::processLine(const std::string& line) {
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

std::map<int32_t, std::pair<std::string, std::string>> BaseFileReader::readFileAndProcessLines(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << fileName << std::endl;
        return {};
    }

    std::map<int32_t, std::pair<std::string, std::string>> result;
    std::string line;
    int32_t index = 0;
    while (std::getline(file, line, '\x1e')) {
        std::map<std::string, std::string> record = processLine(line);

        auto id_itr = record.find("id");
        auto cate_vec_itr = record.find("cate_vec");

        if (id_itr == record.end()) {
            throw std::runtime_error("Error: 'id' not found in line: " + line);
        }

        if (cate_vec_itr == record.end()) {
            throw std::runtime_error("Error: 'cate_vec' not found in line: " + line);
        }

        if (id_itr != record.end() && cate_vec_itr != record.end()) {
            result[index++] = { id_itr->second, cate_vec_itr->second };
        }
    }

    file.close();
    return result;
}