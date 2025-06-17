/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     file.h
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Utility File
 */

#ifndef __MERCURY_UTILITY_FILE_H__
#define __MERCURY_UTILITY_FILE_H__

#include "internal/platform.h"

namespace mercury {

/*! File
 */
class File
{
public:
    //! Constructor
    File(void) : _handle(nullptr), _read_only(false) {}

    //! Constructor
    File(File &&rhs)
    {
        _read_only = rhs._read_only;
        _handle = rhs._handle;
        rhs._read_only = false;
        rhs._handle = nullptr;
    }

    //! Destructor
    ~File(void)
    {
        this->close();
    }

    //! Assignment
    File &operator=(File &&rhs)
    {
        _read_only = rhs._read_only;
        _handle = rhs._handle;
        rhs._read_only = false;
        rhs._handle = nullptr;
        return *this;
    }

    //! Test if the file is valid
    bool isValid(void) const
    {
        return (_handle != nullptr);
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

    //! Reset the file
    void reset(void);

    //! Write data into the file
    size_t write(const void *data, size_t len) const;

    //! Write data into the file
    size_t write(size_t offset, const void *data, size_t len) const;

    //! Read data from the file
    size_t read(void *buf, size_t len) const;

    //! Read data from the file
    size_t read(size_t offset, void *buf, size_t len) const;

    //! Retrieve size of file
    size_t size(void) const;

    //! Delete a name and possibly the file it refers to
    static bool Delete(const char *path);

    //! Change the name or location of a file
    static bool Rename(const char *oldpath, const char *newpath);

    //! Make directories' path
    static bool MakePath(const char *path);

    //! Remove a directory (includes files & subdirectories)
    static bool RemoveDirectory(const char *path);

    //! Remove a file or a directory (includes files & subdirectories)
    static bool RemovePath(const char *path);

    //! Retrieve non-zero if the path is a regular file
    static bool IsRegular(const char *path);

    //! Retrieve non-zero if the path is a directory
    static bool IsDirectory(const char *path);

    //! Retrieve non-zero if the path is a symbolic link
    static bool IsSymbolicLink(const char *path);

    //! Retrieve non-zero if two paths are pointing to the same file
    static bool IsSame(const char *path1, const char *path2);

private:
    //! Disable them
    File(const File &) = delete;
    File &operator=(const File &) = delete;

    //! Members
    void *_handle;
    bool _read_only;
};

} // namespace mercury

#endif // __MERCURY_UTILITY_FILE_H__
