#ifndef __MERCURY_STRINGUTIL_H
#define __MERCURY_STRINGUTIL_H

#include <limits>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include "half/half.hpp"
#include "src/core/framework/index_logger.h"

namespace mercury { namespace core {

class StringUtil
{
public:
    StringUtil();
    ~StringUtil();
public:
    static std::vector<std::string> split(const std::string& text, const std::string &sepStr, bool ignoreEmpty = true);

    static bool strToInt8(const char* str, int8_t& value);
    static bool strToUInt8(const char* str, uint8_t& value);
    static bool strToInt16(const char* str, int16_t& value);
    static bool strToUInt16(const char* str, uint16_t& value);
    static bool strToInt32(const char* str, int32_t& value);
    static bool strToUInt32(const char* str, uint32_t& value);
    static bool strToInt64(const char* str, int64_t& value);
    static bool strToUInt64(const char* str, uint64_t& value);
    static bool strToHalf(const char* str, half_float::half& value);
    static bool strToFloat(const char *str, float &value);
    static bool strToDouble(const char *str, double &value);

    static std::string vectorToStr(const std::vector<float>& float_vec);

    template<typename T>
    static T fromString(const std::string &str);

    template<typename T>
    static bool fromString(const std::string &str, T &value);

    template<typename T>
    static void fromString(const std::vector<std::string> &strVec, std::vector<T> &vec);

    template<typename T>
    static void fromString(const std::string &str, std::vector<T> &vec, const std::string &delim);

    template<typename T>
    static void fromString(const std::string &str, std::vector<std::vector<T> > &vec, const std::string &delim, const std::string &delim2);

};

template<typename T>
inline T StringUtil::fromString(const std::string& str) {
    T value = T();
    fromString(str, value);
    return value;
}

template<>
inline bool StringUtil::fromString(const std::string& str, std::string &value) {
    value = str;
    return true;
}

template<>
inline bool StringUtil::fromString(const std::string& str, int8_t &value) {
    bool ret = strToInt8(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, uint8_t &value) {
    bool ret = strToUInt8(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, int16_t &value) {
    bool ret = strToInt16(str.data(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, uint16_t &value) {
    bool ret = strToUInt16(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, int32_t &value) {
    bool ret = strToInt32(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, uint32_t &value) {
    bool ret = strToUInt32(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, int64_t &value) {
    bool ret = strToInt64(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, uint64_t &value) {
    bool ret = strToUInt64(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, float &value) {
    bool ret = strToFloat(str.c_str(), value);
    return ret;
}

template<>
inline bool StringUtil::fromString(const std::string& str, double &value) {
    bool ret = strToDouble(str.c_str(), value);
    return ret;
}

template<typename T>
inline void StringUtil::fromString(const std::vector<std::string> &strVec, std::vector<T> &vec) {
    vec.clear();
    vec.reserve(strVec.size());
    for (uint32_t i = 0; i < strVec.size(); ++i) {
        vec.push_back(fromString<T>(strVec[i]));
    }
}

template<typename T>
inline void StringUtil::fromString(const std::string &str, std::vector<T> &vec, const std::string &delim) {
    std::vector<std::string> strVec = split(str, delim);
    fromString(strVec, vec);
}

template<typename T>
inline void StringUtil::fromString(const std::string &str, std::vector<std::vector<T> > &vec, const std::string &delim1, const std::string &delim2) {
    vec.clear();
    std::vector<std::string> strVec;
    fromString(str, strVec, delim2);
    vec.resize(strVec.size());
    for (uint32_t i = 0; i < strVec.size(); ++i) {
        fromString(strVec[i], vec[i], delim1);
    }
}

}}; //namespace mercury

#endif //__MERCURY_STRINGUTIL_H
