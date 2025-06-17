/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mmap_file.cc
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Utility Memory Map File
 */

#include "mmap_file.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h> 
#include "framework/index_logger.h"

namespace mercury {

static inline void CloseSafely(int fd)
{
    int ret = ::close(fd);
    while (ret == -1 && errno == EINTR) {
        ret = ::close(fd);
    }
}

bool MMapFile::create(const char *path, size_t size)
{
    platform_false_if_false(!_region);

    // Try opening or creating a file
    int fd =
        ::open(path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    platform_false_if_lt_zero(fd);

    // Truncate the file to the specified size
    platform_do_if_ne_zero(::ftruncate(fd, size))
    {
        CloseSafely(fd);
        return false;
    }

    _read_only = false;
    _region_size = size;
    _region = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    platform_do_if_false(_region != MAP_FAILED)
    {
        CloseSafely(fd);
        _region = nullptr;
        return false;
    }

    // Closing the file descriptor does not unmap the region
    CloseSafely(fd);
    return true;
}

bool MMapFile::open(const char *path, bool rdonly)
{
    platform_false_if_false(!_region);

    // Try opening the file
    int fd = ::open(path, (rdonly ? O_RDONLY : O_RDWR),
                    S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    platform_false_if_lt_zero(fd);

    // Get stat of file
    struct stat fs;
    platform_do_if_ne_zero(::fstat(fd, &fs))
    {
        CloseSafely(fd);
        return false;
    }

    int prot = rdonly ? PROT_READ : PROT_READ | PROT_WRITE;
    _read_only = rdonly;
    _region_size = fs.st_size;
    _region = ::mmap(nullptr, fs.st_size, prot, MAP_SHARED | MAP_LOCKED, fd, 0);
    if(_region == MAP_FAILED){
        LOG_ERROR("mmap lock error %d, error msg %s, try use unlocked flag!", errno, strerror(errno));
        _region = ::mmap(nullptr, fs.st_size, prot, MAP_SHARED, fd, 0);
    }

    platform_do_if_false(_region != MAP_FAILED)
    {
        CloseSafely(fd);
        _region = nullptr;
        return false;
    }

    // Closing the file descriptor does not unmap the region
    CloseSafely(fd);
    return true;
}

void MMapFile::close(void)
{
    if (_region) {
        ::munmap(_region, _region_size);
        _region = nullptr;
    }
}

bool MMapFile::flush(bool block)
{
    platform_false_if_false(_region);
    return (::msync(_region, _region_size, block ? MS_SYNC : MS_ASYNC) == 0);
}

} // namespace mercury
