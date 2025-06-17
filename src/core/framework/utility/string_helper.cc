/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     string_helper.cc
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Utility String Helper
 */

#include "string_helper.h"
#include <utility>

namespace mercury {

static std::string StringMove(std::string &&str)
{
    return std::move(str);
}

static double StringToDouble(std::string &&str)
{
    return std::strtod(str.c_str(), nullptr);
}

static float StringToFloat(std::string &&str)
{
    return std::strtof(str.c_str(), nullptr);
}

static int8_t StringToInt8(std::string &&str)
{
    return static_cast<int8_t>(std::strtol(str.c_str(), nullptr, 0));
}

static int16_t StringToInt16(std::string &&str)
{
    return static_cast<int16_t>(std::strtol(str.c_str(), nullptr, 0));
}

static int32_t StringToInt32(std::string &&str)
{
    return static_cast<int32_t>(std::strtol(str.c_str(), nullptr, 0));
}

static int64_t StringToInt64(std::string &&str)
{
    return static_cast<int64_t>(std::strtoll(str.c_str(), nullptr, 0));
}

static uint8_t StringToUint8(std::string &&str)
{
    return static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 0));
}

static uint16_t StringToUint16(std::string &&str)
{
    return static_cast<uint16_t>(std::strtoul(str.c_str(), nullptr, 0));
}

static uint32_t StringToUint32(std::string &&str)
{
    return static_cast<uint32_t>(std::strtoul(str.c_str(), nullptr, 0));
}

static uint64_t StringToUint64(std::string &&str)
{
    return static_cast<uint64_t>(std::strtoull(str.c_str(), nullptr, 0));
}

template <typename T, T (*TFunc)(std::string &&)>
static void StringSplit(const std::string &str, char delim, std::vector<T> *out)
{
    out->clear();

    size_t a = 0, b = str.find(delim);
    while (b != std::string::npos) {
        out->push_back(TFunc(str.substr(a, b - a)));
        a = b + 1;
        b = str.find(delim, a);
    }
    out->push_back(TFunc(str.substr(a, str.length() - a)));
}

template <typename T, T (*TFunc)(std::string &&)>
static void StringSplit(const std::string &str, const std::string &delim,
                        std::vector<T> *out)
{
    out->clear();

    if (!delim.empty()) {
        size_t a = 0, b = str.find(delim);
        while (b != std::string::npos) {
            out->push_back(TFunc(str.substr(a, b - a)));
            a = b + delim.length();
            b = str.find(delim, a);
        }
        out->push_back(TFunc(str.substr(a, str.length() - a)));
    } else {
        out->push_back(TFunc(str.substr()));
    }
}

bool StringHelper::StartsWith(const std::string &ref, const std::string &prefix)
{
    return (ref.size() >= prefix.size()) &&
           (ref.compare(0, prefix.size(), prefix) == 0);
}

bool StringHelper::EndsWith(const std::string &ref, const std::string &suffix)
{
    size_t s1 = ref.size();
    size_t s2 = suffix.size();
    return (s1 >= s2) && (ref.compare(s1 - s2, s2, suffix) == 0);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<std::string> *out)
{
    StringSplit<std::string, StringMove>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<std::string> *out)
{
    StringSplit<std::string, StringMove>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<float> *out)
{
    StringSplit<float, StringToFloat>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<float> *out)
{
    StringSplit<float, StringToFloat>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<double> *out)
{
    StringSplit<double, StringToDouble>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<double> *out)
{
    StringSplit<double, StringToDouble>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<int8_t> *out)
{
    StringSplit<int8_t, StringToInt8>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<int8_t> *out)
{
    StringSplit<int8_t, StringToInt8>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<uint8_t> *out)
{
    StringSplit<uint8_t, StringToUint8>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<uint8_t> *out)
{
    StringSplit<uint8_t, StringToUint8>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<int16_t> *out)
{
    StringSplit<int16_t, StringToInt16>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<int16_t> *out)
{
    StringSplit<int16_t, StringToInt16>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<uint16_t> *out)
{
    StringSplit<uint16_t, StringToUint16>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<uint16_t> *out)
{
    StringSplit<uint16_t, StringToUint16>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<int32_t> *out)
{
    StringSplit<int32_t, StringToInt32>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<int32_t> *out)
{
    StringSplit<int32_t, StringToInt32>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<uint32_t> *out)
{
    StringSplit<uint32_t, StringToUint32>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<uint32_t> *out)
{
    StringSplit<uint32_t, StringToUint32>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<int64_t> *out)
{
    StringSplit<int64_t, StringToInt64>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<int64_t> *out)
{
    StringSplit<int64_t, StringToInt64>(str, delim, out);
}

void StringHelper::Split(const std::string &str, char delim,
                         std::vector<uint64_t> *out)
{
    StringSplit<uint64_t, StringToUint64>(str, delim, out);
}

void StringHelper::Split(const std::string &str, const std::string &delim,
                         std::vector<uint64_t> *out)
{
    StringSplit<uint64_t, StringToUint64>(str, delim, out);
}

} // namespace mercury
