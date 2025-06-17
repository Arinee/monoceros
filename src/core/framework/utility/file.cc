/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     mmap_file.cc
 *   \author   Hechong.xyf
 *   \date     Nov 2017
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Utility File
 */

#include "file.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mercury {

char *JoinFilePath(const char *prefix, const char *suffix)
{
    size_t prefix_len = strlen(prefix);
    size_t suffix_len = strlen(suffix);

    char *path = (char *)malloc(prefix_len + suffix_len + 2);
    if (path) {
        memcpy(path, prefix, prefix_len);
        memcpy(path + prefix_len + 1, suffix, suffix_len);
        path[prefix_len] = '/';
        path[prefix_len + suffix_len + 1] = '\0';
    }
    return path;
}

static inline int OpenSafely(const char *path, int flags)
{
    int fd = open(path, flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    while (fd == -1 && errno == EINTR) {
        fd = open(path, flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    }
    return fd;
}

static inline void CloseSafely(int fd)
{
    int ret = close(fd);
    while (ret == -1 && errno == EINTR) {
        ret = close(fd);
    }
}

static inline ssize_t ReadSafely(int fd, void *buf, size_t count)
{
    ssize_t ret = read(fd, buf, count);
    while (ret == -1 && errno == EINTR) {
        ret = read(fd, buf, count);
    }
    return ret;
}

static inline ssize_t PreadSafely(int fd, void *buf, size_t count,
                                  size_t offset)
{
    ssize_t ret = pread(fd, buf, count, offset);
    while (ret == -1 && errno == EINTR) {
        ret = pread(fd, buf, count, offset);
    }
    return ret;
}

static inline ssize_t WriteSafely(int fd, const void *buf, size_t count)
{
    ssize_t ret = write(fd, buf, count);
    while (ret == -1 && errno == EINTR) {
        ret = write(fd, buf, count);
    }
    return ret;
}

static inline ssize_t PwriteSafely(int fd, const void *buf, size_t count,
                                   size_t offset)
{
    ssize_t ret = pwrite(fd, buf, count, offset);
    while (ret == -1 && errno == EINTR) {
        ret = pwrite(fd, buf, count, offset);
    }
    return ret;
}

static inline size_t ReadAll(int fd, void *buf, size_t count)
{
    size_t rdlen = 0;
    while (rdlen < count) {
        ssize_t ret = ReadSafely(fd, (char *)buf + rdlen, count - rdlen);
        if (ret <= 0) {
            break;
        }
        rdlen += ret;
    }
    return rdlen;
}

static inline size_t PreadAll(int fd, void *buf, size_t count, size_t offset)
{
    size_t rdlen = 0;
    while (rdlen < count) {
        ssize_t ret =
            PreadSafely(fd, (char *)buf + rdlen, count - rdlen, offset + rdlen);
        if (ret <= 0) {
            break;
        }
        rdlen += ret;
    }
    return rdlen;
}

static inline size_t WriteAll(int fd, const void *buf, size_t count)
{
    size_t wrlen = 0;
    while (wrlen < count) {
        ssize_t ret = WriteSafely(fd, (const char *)buf + wrlen, count - wrlen);
        if (ret <= 0) {
            break;
        }
        wrlen += ret;
    }
    return wrlen;
}

static inline size_t PwriteAll(int fd, const void *buf, size_t count,
                               size_t offset)
{
    size_t wrlen = 0;
    while (wrlen < count) {
        ssize_t ret = PwriteSafely(fd, (const char *)buf + wrlen, count - wrlen,
                                   offset + wrlen);
        if (ret <= 0) {
            break;
        }
        wrlen += ret;
    }
    return wrlen;
}

bool File::create(const char *path, size_t len)
{
    platform_false_if_false(!_handle);

    // Try opening or creating a file
    int fd = OpenSafely(path, O_RDWR | O_CREAT);
    platform_false_if_lt_zero(fd);

    // Truncate the file to the specified size
    platform_do_if_ne_zero(ftruncate(fd, len))
    {
        CloseSafely(fd);
        return false;
    }

    _read_only = false;
    _handle = (void *)(intptr_t)fd;
    return true;
}

bool File::open(const char *path, bool rdonly)
{
    platform_false_if_false(!_handle);

    // Try opening the file
    int fd = OpenSafely(path, (rdonly ? O_RDONLY : O_RDWR));
    platform_false_if_lt_zero(fd);

    _read_only = rdonly;
    _handle = (void *)(intptr_t)fd;
    return true;
}

void File::close(void)
{
    platform_return_if_false(_handle);
    CloseSafely((int)(intptr_t)_handle);
    _handle = nullptr;
}

void File::reset(void)
{
    platform_return_if_false(_handle);
    lseek((int)(intptr_t)_handle, 0, SEEK_SET);
}

size_t File::write(const void *data, size_t len) const
{
    return WriteAll((int)(intptr_t)_handle, data, len);
}

size_t File::write(size_t offset, const void *data, size_t len) const
{
    return PwriteAll((int)(intptr_t)_handle, data, len, offset);
}

size_t File::read(void *buf, size_t len) const
{
    return ReadAll((int)(intptr_t)_handle, buf, len);
}

size_t File::read(size_t offset, void *buf, size_t len) const
{
    return PreadAll((int)(intptr_t)_handle, buf, len, offset);
}

size_t File::size(void) const
{
    struct stat fs;
    platform_zero_if_false(_handle && fstat((int)(intptr_t)_handle, &fs) == 0);
    return (fs.st_size);
}

bool File::Delete(const char *path)
{
    // Delete a file by the path
    return (unlink(path) == 0);
}

bool File::Rename(const char *oldpath, const char *newpath)
{
    return (rename(oldpath, newpath) == 0);
}

bool File::MakePath(const char *path)
{
    char pathbuf[PATH_MAX];
    char *sp, *pp;

    strncpy(pathbuf, path, sizeof(pathbuf) - 1);
    pathbuf[PATH_MAX - 1] = '\0';

    pp = pathbuf;
    while ((sp = strchr(pp, '/')) != NULL) {
        // Neither root nor double slash in path
        if (sp != pp) {
            *sp = '\0';
            if (mkdir(pathbuf, 0755) == -1 && errno != EEXIST) {
                return false;
            }
            *sp = '/';
        }
        pp = sp + 1;
    }
    return !(*pp != '\0' && mkdir(pathbuf, 0755) == -1 && errno != EEXIST);
}

bool File::RemoveDirectory(const char *path)
{
    DIR *dir = opendir(path);
    if (!dir) {
        return false;
    }

    struct dirent *dent;
    while ((dent = readdir(dir)) != nullptr) {
        if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) {
            continue;
        }
        char *fullpath = JoinFilePath(path, dent->d_name);
        if (!fullpath) {
            continue;
        }

        if (File::IsDirectory(fullpath)) {
            File::RemoveDirectory(fullpath);
        } else {
            unlink(fullpath);
        }
        free(fullpath);
    }
    closedir(dir);
    return (rmdir(path) == 0);
}

bool File::RemovePath(const char *path)
{
    if (File::IsDirectory(path)) {
        return File::RemoveDirectory(path);
    }
    return (unlink(path) == 0);
}

bool File::IsRegular(const char *path)
{
    struct stat buf;
    if (stat(path, &buf) != 0) {
        return false;
    }
    return ((buf.st_mode & S_IFREG) != 0);
}

bool File::IsDirectory(const char *path)
{
    struct stat buf;
    if (stat(path, &buf) != 0) {
        return false;
    }
    return ((buf.st_mode & S_IFDIR) != 0);
}

bool File::IsSymbolicLink(const char *path)
{
    struct stat buf;
    if (stat(path, &buf) != 0) {
        return false;
    }
    return ((buf.st_mode & S_IFLNK) != 0);
}

bool File::IsSame(const char *path1, const char *path2)
{
    char real_path1[PATH_MAX];
    char real_path2[PATH_MAX];
    if (!::realpath(path1, real_path1)) {
        return false;
    }
    if (!::realpath(path2, real_path2)) {
        return false;
    }
    return (!::strcmp(real_path1, real_path2));
}

} // namespace mercury
