/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     string_helper.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility String Helper
 */

#ifndef __MERCURY_UTILITY_STRING_HELPER_H__
#define __MERCURY_UTILITY_STRING_HELPER_H__

#include "internal/platform.h"
#include <string>
#include <vector>

namespace mercury {

/*! String Helper
 */
struct StringHelper
{
    //! Return true if the `ref` starts with the given prefix
    static bool StartsWith(const std::string &ref, const std::string &prefix);

    //! Return true if the `ref` ends with the given suffix
    static bool EndsWith(const std::string &ref, const std::string &suffix);

    //! Split a string into an array of substrings
    static void Split(const std::string &str, char delim,
                      std::vector<std::string> *out);

    //! Split a string into an array of substrings
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<std::string> *out);

    //! Split a string into an array of substrings (float)
    static void Split(const std::string &str, char delim,
                      std::vector<float> *out);

    //! Split a string into an array of substrings (float)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<float> *out);

    //! Split a string into an array of substrings (double)
    static void Split(const std::string &str, char delim,
                      std::vector<double> *out);

    //! Split a string into an array of substrings (double)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<double> *out);

    //! Split a string into an array of substrings (int8_t)
    static void Split(const std::string &str, char delim,
                      std::vector<int8_t> *out);

    //! Split a string into an array of substrings (int8_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<int8_t> *out);

    //! Split a string into an array of substrings (uint8_t)
    static void Split(const std::string &str, char delim,
                      std::vector<uint8_t> *out);

    //! Split a string into an array of substrings (uint8_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<uint8_t> *out);

    //! Split a string into an array of substrings (int16_t)
    static void Split(const std::string &str, char delim,
                      std::vector<int16_t> *out);

    //! Split a string into an array of substrings (int16_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<int16_t> *out);

    //! Split a string into an array of substrings (uint16_t)
    static void Split(const std::string &str, char delim,
                      std::vector<uint16_t> *out);

    //! Split a string into an array of substrings (uint16_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<uint16_t> *out);

    //! Split a string into an array of substrings (int32_t)
    static void Split(const std::string &str, char delim,
                      std::vector<int32_t> *out);

    //! Split a string into an array of substrings (int32_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<int32_t> *out);

    //! Split a string into an array of substrings (uint32_t)
    static void Split(const std::string &str, char delim,
                      std::vector<uint32_t> *out);

    //! Split a string into an array of substrings (uint32_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<uint32_t> *out);

    //! Split a string into an array of substrings (int64_t)
    static void Split(const std::string &str, char delim,
                      std::vector<int64_t> *out);

    //! Split a string into an array of substrings (int64_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<int64_t> *out);

    //! Split a string into an array of substrings (uint64_t)
    static void Split(const std::string &str, char delim,
                      std::vector<uint64_t> *out);

    //! Split a string into an array of substrings (uint64_t)
    static void Split(const std::string &str, const std::string &delim,
                      std::vector<uint64_t> *out);
};

} // namespace mercury

#endif // __MERCURY_UTILITY_STRING_HELPER_H__
