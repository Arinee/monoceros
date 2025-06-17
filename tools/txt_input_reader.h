#ifndef __TXT_INPUT_READER_H__
#define __TXT_INPUT_READER_H__

#include "utils/string_util.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace mercury {

// support type: float, binary, int16, int8
template <typename T>
class TxtInputReader {
public:
    bool loadQuery(const std::string &queryFile, 
                   const std::string &firstSep,
                   const std::string &secondSep,
                   std::vector<std::vector<T>> &features)
    {
        std::fstream qf(queryFile, std::ios::in);

        if (!qf.is_open()) {
            std::cerr << "open query file failed! [" << queryFile << "]" << std::endl;
            return false;
        }

        std::string buffer;
        while (getline(qf, buffer)) {
            buffer.erase(buffer.find_last_not_of('\n') + 1);
            if (buffer.empty()) {
                continue;
            }
            std::vector<std::string> &&res = StringUtil::split(buffer, firstSep);
            if (res.empty()) {
                continue;
            }
            std::string featureStr = res[0];
            if (res.size() > 1) {
                featureStr = res[1];
            }
            std::vector<T> feature;
            size_t dimension = 0;
            loadFromString(featureStr, secondSep, feature, &dimension);
            features.emplace_back(feature);
        }
        qf.close();
        if (features.size() == 0) {
            std::cerr << "Read query size is 0" << std::endl;
            return false;
        }
        std::cout << "Read query dimension: " << features[0].size() << std::endl;
        return true;
    }

    bool loadRecord(const std::string &input, 
                    const std::string &firstSep,
                    const std::string &secondSep,
                    const size_t dimension,
                    std::vector<std::vector<T>> &features,
                    std::vector<uint64_t> &keys)
    {
        std::fstream qf(input, std::ios::in);

        if (!qf.is_open()) {
            std::cerr << "open file failed! [" 
                << input << "]" << std::endl;
            return false;
        }

        std::string buffer;
        while (getline(qf, buffer)) {
            buffer.erase(buffer.find_last_not_of('\n') + 1);
            if (buffer.empty()) {
                continue;
            }
            std::vector<std::string> &&res = StringUtil::split(buffer, firstSep);
            if (res.size() < 2) {
                std::cerr << "skip record : " << buffer << std::endl;
                continue;
            }
            std::vector<T> feature;
            size_t realDim = 0;
            loadFromString(res[1], secondSep, feature, &realDim);
            if (realDim != dimension) {
                std::cerr << "real dim (" << realDim
                    << ") is not equal to dimension(" << dimension 
                    << ") key : " << res[0]
                    << std::endl;
                continue;
            }
            features.emplace_back(feature);
            keys.emplace_back(atol(res[0].c_str()));
        }
        qf.close();
        if (keys.size() == 0) {
            std::cerr << "Reading nothing from input" << std::endl;
            return false;
        }
        std::cout << "Read Record dimension: " << features[0].size() << std::endl;
        return true;
    }

    template<typename U>
    void loadFromString(const std::string &record, 
                        const std::string &secondSep,
                        std::vector<U> &feature,
                        size_t *dimension)
    {
        StringUtil::fromString(record, feature, secondSep);
        *dimension = feature.size();
    }

    // overloading for binary 
    void loadFromString(const std::string &record, 
                        const std::string &secondSep,
                        std::vector<uint32_t> &feature,
                        size_t *dimension)
    {
        // fetch split value from text file
        std::vector<uint8_t> vec;
        StringUtil::fromString(record, vec, secondSep);
        if (vec.size() == 0) {
            std::cerr << "Binary vector size is 0" << std::endl;
            return;
        }
        if (vec.size() % 32 != 0) {
            std::cerr << "Binary vector size must be 32X" << std::endl;
            return;
        }
        // compact into uint32_t
        size_t sz = vec.size();
        std::vector<uint8_t> tmp;
        for (size_t i = 0; i < sz; i += 8) {
            uint8_t v = 0;
            v |= (vec[i  ] & 0x01) << 7;
            v |= (vec[i+1] & 0x01) << 6;
            v |= (vec[i+2] & 0x01) << 5;
            v |= (vec[i+3] & 0x01) << 4;
            v |= (vec[i+4] & 0x01) << 3;
            v |= (vec[i+5] & 0x01) << 2;
            v |= (vec[i+6] & 0x01) << 1;
            v |= (vec[i+7] & 0x01) << 0;
            tmp.push_back(v);
        }
        feature.resize(sz/32);
        memcpy(&feature[0], &tmp[0], tmp.size());
        *dimension = sz;
    }
};


}; // namespace mercury


#endif // __TXT_INPUT_READER_H__
