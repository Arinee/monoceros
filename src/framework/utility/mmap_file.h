/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mmap_file.h
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility Memory Mapping File
 */

#ifndef __MERCURY_UTILITY_MMAP_FILE_H__
#define __MERCURY_UTILITY_MMAP_FILE_H__

#include "internal/platform.h"

namespace mercury {

/*! Memory Mapping File
 */
class MMapFile
{
public:
    //! Constructor
    MMapFile(void) : _read_only(false), _region(nullptr), _region_size(0) {}

    //! Constructor
    MMapFile(MMapFile &&rhs)
    {
        _read_only = rhs._read_only;
        _region = rhs._region;
        _region_size = rhs._region_size;
        rhs._read_only = false;
        rhs._region = nullptr;
        rhs._region_size = 0;
    }

    //! Destructor
    ~MMapFile(void)
    {
        this->close();
    }

    //! Assignment
    MMapFile &operator=(MMapFile &&rhs)
    {
        _read_only = rhs._read_only;
        _region = rhs._region;
        _region_size = rhs._region_size;
        rhs._read_only = false;
        rhs._region = nullptr;
        rhs._region_size = 0;
        return *this;
    }

    //! Test if the file is valid
    bool isValid(void) const
    {
        return (_region != nullptr);
    }

    //! Retrieve non-zero if memory region is read only
    bool isReadOnly(void) const
    {
        return _read_only;
    }

    //! Create a memory mapping file
    bool create(const char *path, size_t size);

    //! Open a memory mapping file
    bool open(const char *path, bool rdonly);

    //! Close the memory mapping file
    void close(void);

    //! Synchronize memory with physical storage
    bool flush(bool block);

    //! Retrieve memory region of file
    void *region(void) const
    {
        return _region;
    }

    //! Retrieve region size of file
    size_t region_size(void) const
    {
        return _region_size;
    }

private:
    //! Disable them
    MMapFile(const MMapFile &) = delete;
    MMapFile &operator=(const MMapFile &) = delete;

    //! Members
    bool _read_only;
    void *_region;
    size_t _region_size;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_MMAP_FILE_H__
