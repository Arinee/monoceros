/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     crc32c.h
 *   \author   Hechong.xyf
 *   \date     Dec 2017
 *   \version  1.0.0
 *   \brief    Interface of CRC32C (Castagnoli)
 */

#ifndef __MERCURY_UTILITY_INTERNAL_CRC32C_H__
#define __MERCURY_UTILITY_INTERNAL_CRC32C_H__

#include "platform.h"

namespace mercury {
namespace internal {

//! Compute the CRC32C checksum for the source data buffer
uint32_t Crc32c(uint32_t crc, const void *data, size_t len);

} // namespace internal
} // namespace mercury

#endif // __MERCURY_UTILITY_INTERNAL_CRC32C_H__
