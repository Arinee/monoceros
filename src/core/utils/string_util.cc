#include "string_util.h"
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <algorithm>
#include <sstream>

using namespace std;
namespace mercury { namespace core {

StringUtil::StringUtil() { 
}

StringUtil::~StringUtil() { 
}

std::vector<std::string> StringUtil::split(const std::string& text, 
        const std::string &sepStr, bool ignoreEmpty)
{
    std::vector<std::string> vec;
    std::string str(text);
    std::string sep(sepStr);
    size_t n = 0, old = 0;
    while (n != std::string::npos)
    {
        n = str.find(sep,n);
        if (n != std::string::npos)
        {
            if (!ignoreEmpty || n != old) 
                vec.push_back(str.substr(old, n-old));
            n += sep.length();
            old = n;
        }
    }

    if (!ignoreEmpty || old < str.length()) {
        vec.push_back(str.substr(old, str.length() - old));
    }
    return vec;
}

std::string StringUtil::vectorToStr(const std::vector<float>& float_vec) {
    ostringstream ss;
    for (size_t i = 0; i < float_vec.size(); i++) {
        ss << float_vec.at(i);
        if (i != float_vec.size() - 1) {
            ss << " ";
        }
    }

    return ss.str();
}

bool StringUtil::strToInt8(const char* str, int8_t& value)
{
    int32_t v32 = 0;
    bool ret = strToInt32(str, v32);
    value = (int8_t)v32;
        
    return ret && v32 >= INT8_MIN && v32 <= INT8_MAX;
}

bool StringUtil::strToUInt8(const char* str, uint8_t& value)
{
    uint32_t v32 = 0;
    bool ret = strToUInt32(str, v32);
    value = (uint8_t)v32;

    return ret && v32 <= UINT8_MAX;
}

bool StringUtil::strToInt16(const char* str, int16_t& value)
{
    int32_t v32 = 0;
    bool ret = strToInt32(str, v32);
    value = (int16_t)v32;    
    return ret && v32 >= INT16_MIN && v32 <= INT16_MAX;
}

bool StringUtil::strToUInt16(const char* str, uint16_t& value)
{
    uint32_t v32 = 0;
    bool ret = strToUInt32(str, v32);
    value = (uint16_t)v32;
    return ret && v32 <= UINT16_MAX;
}

bool StringUtil::strToInt32(const char* str, int32_t& value) 
{
    if (NULL == str || *str == 0) 
    {
        return false;
    }
    char* endPtr = NULL;
    errno = 0;

# if __WORDSIZE == 64
    int64_t value64 = strtol(str, &endPtr, 10);
    if (value64 < INT32_MIN || value64 > INT32_MAX)
    {
        return false;
    }
    value = (int32_t)value64;
# else
    value = (int32_t)strtol(str, &endPtr, 10);
# endif

    if (errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

bool StringUtil::strToUInt32(const char* str, uint32_t& value) 
{
    if (NULL == str || *str == 0 || *str == '-') 
    {
        return false;
    }
    char* endPtr = NULL;
    errno = 0;

# if __WORDSIZE == 64
    uint64_t value64 = strtoul(str, &endPtr, 10);
    if (value64 > UINT32_MAX)
    {
        return false;
    }
    value = (int32_t)value64;
# else
    value = (uint32_t)strtoul(str, &endPtr, 10);
# endif

    if (errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

bool StringUtil::strToUInt64(const char* str, uint64_t& value)
{
    if (NULL == str || *str == 0 || *str == '-') 
    {
        return false;
    }
    char* endPtr = NULL;
    errno = 0;
    value = (uint64_t)strtoull(str, &endPtr, 10);
    if (errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

bool StringUtil::strToInt64(const char* str, int64_t& value) 
{
    if (NULL == str || *str == 0) 
    {
        return false;
    }
    char* endPtr = NULL;
    errno = 0;
    value = (int64_t)strtoll(str, &endPtr, 10);
    if (errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

bool StringUtil::strToHalf(const char* str, half_float::half& value)
{
    if (NULL == str || *str == 0) 
    {
        return false;
    }
    errno = 0;
    char* endPtr = NULL;
    float f = strtof(str, &endPtr);
    if (!std::isnan(f) && errno == 0 && endPtr && *endPtr == 0) 
    {
        value = half_float::half_cast<half_float::half, float>(f);
        return true;
    }
    return false;
}

bool StringUtil::strToFloat(const char* str, float& value) 
{
    if (NULL == str || *str == 0) 
    {
        return false;
    }
    errno = 0;
    char* endPtr = NULL;
    value = strtof(str, &endPtr);
    if (!std::isnan(value) && errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

bool StringUtil::strToDouble(const char* str, double& value) 
{
    if (NULL == str || *str == 0) 
    {
        return false;
    }
    errno = 0;
    char* endPtr = NULL;
    value = strtod(str, &endPtr);
    if (!std::isnan(value) && errno == 0 && endPtr && *endPtr == 0) 
    {
        return true;
    }
    return false;
}

}; // namespace core
}; // namespace mercury

