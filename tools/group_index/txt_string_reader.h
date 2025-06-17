#ifndef __TXT_INPUT_READER_H__
#define __TXT_INPUT_READER_H__

#include "src/core/utils/string_util.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace mercury {
using namespace mercury::core;
// support type: float, binary, int16, int8
template <typename T>
class TxtStringReader {
public:
    bool loadQuery(const std::string &queryFile, 
                   const std::string &firstSep,
                   std::vector<std::string> &features)
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
            features.emplace_back(featureStr);
        }
        qf.close();
        if (features.size() == 0) {
            std::cerr << "Read query size is 0" << std::endl;
            return false;
        }
        std::cout << "Read query done: " << std::endl;
        return true;
    }
};


}; // namespace mercury


#endif // __TXT_INPUT_READER_H__
