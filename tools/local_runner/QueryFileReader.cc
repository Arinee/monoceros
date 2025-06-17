#include "QueryFileReader.h"

QueryFileReader::QueryFileReader(const std::string& filename) : filename_(filename) {}

std::vector<std::pair<std::string, std::string>> QueryFileReader::getRequests() {
    std::vector<std::pair<std::string, std::string>> requests;
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filename_);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::string::size_type id_start = line.find("\"request_id\":");
        std::string::size_type url_start = line.find("\"request_url\":");

        if (id_start == std::string::npos) {
            throw std::runtime_error("Error: 'request_id' not found in line: " + line);
        }

        if (url_start == std::string::npos) {
            throw std::runtime_error("Error: 'request_url' not found in line: " + line);
        }

        if (id_start != std::string::npos && url_start != std::string::npos) {
            id_start = line.find_first_of('"', id_start + 13);
            std::string::size_type id_end = line.find_first_of('"', id_start + 1);
            url_start = line.find_first_of('"', url_start + 13);
            std::string::size_type url_end = line.find_first_of('"', url_start + 1);

            if (id_start != std::string::npos && id_end != std::string::npos &&
                url_start != std::string::npos && url_end != std::string::npos) {
                std::string request_id = line.substr(id_start + 1, id_end - id_start - 1);
                std::string request_url = line.substr(url_start + 1, url_end - url_start - 1);
                requests.emplace_back(std::make_pair(request_id, request_url));
            }
        }
    }
    file.close();
    return requests;
}
