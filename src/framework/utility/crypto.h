/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     crypto.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Crypto
 */

#ifndef __MERCURY_UTILITY_CRYPTO_H__
#define __MERCURY_UTILITY_CRYPTO_H__

#include "internal/crc32c.h"
#include <cstring>
#include <string>

namespace mercury {

/*! Crypto module
 */
struct Crypto
{
    //! Compute the CRC32C checksum for the source data buffer
    static uint32_t Crc32c(uint32_t crc, const void *data, size_t len)
    {
        return internal::Crc32c(crc, data, len);
    }

    //! Compute the CRC32C checksum for the source data buffer
    static uint32_t Crc32c(const void *data, size_t len)
    {
        return Crypto::Crc32c(0, data, len);
    }

    //! Compute the CRC32C checksum for a STL string
    static uint32_t Crc32c(uint32_t crc, const std::string &str)
    {
        return Crypto::Crc32c(crc, str.data(), str.size());
    }

    //! Compute the CRC32C checksum for a STL string
    static uint32_t Crc32c(const std::string &str)
    {
        return Crypto::Crc32c(0, str.data(), str.size());
    }

    //! Compute the CRC32C checksum for a C string
    static uint32_t Crc32c(uint32_t crc, const char *str)
    {
        return Crypto::Crc32c(crc, str, std::strlen(str));
    }

    //! Compute the CRC32C checksum for a C string
    static uint32_t Crc32c(const char *str)
    {
        return Crypto::Crc32c(0, str, std::strlen(str));
    }
};

} // namespace mercury

#endif // __MERCURY_UTILITY_CRYPTO_H__
