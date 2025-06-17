#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include "src/core/common/common.h"
#include "src/core/framework/index_framework.h"

MERCURY_NAMESPACE_BEGIN(core);

#define BUFFER_SIZE_LARGE_FOR_CACHED_IO (size_t)1024 * (size_t)1048576

#define BUFFER_SIZE_MID_FOR_CACHED_IO (size_t)128 * (size_t)1048576

#define BUFFER_SIZE_SMALL_FOR_CACHED_IO (size_t)64 * (size_t)1048576

// sequential cached reads
class cached_ifstream
{
  public:
    cached_ifstream()
    {
    }
    cached_ifstream(const std::string &filename, uint64_t cacheSize) : cache_size(cacheSize), cur_off(0)
    {
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        this->open(filename, cache_size);
    }
    ~cached_ifstream()
    {
        delete[] cache_buf;
        reader.close();
    }

    void open(const std::string &filename, uint64_t cacheSize)
    {
        this->cur_off = 0;

        try
        {
            reader.open(filename, std::ios::binary | std::ios::ate);
            fsize = reader.tellg();
            reader.seekg(0, std::ios::beg);
            if (!reader.is_open()) {
                throw std::io_errc();
            }

            if (cacheSize == 0) {
                throw std::runtime_error("cacheSize is 0");
            }
            cacheSize = (std::min)(cacheSize, fsize);
            this->cache_size = cacheSize;
            cache_buf = new char[cacheSize];
            reader.read(cache_buf, cacheSize);
            LOG_INFO("Opened: %s, size: %lu, cache_size: %lu", filename.c_str(), fsize, cacheSize);
        }
        catch (std::system_error &e)
        {
            LOG_ERROR("io_error");
            throw e;
        }
    }

    size_t get_file_size()
    {
        return fsize;
    }

    void read(char *read_buf, uint64_t n_bytes)
    {
        if (cache_buf == nullptr) {
            throw std::runtime_error("cache_buf is null");
        }

        if (read_buf == nullptr) {
            throw std::io_errc();
        }

        if (n_bytes <= (cache_size - cur_off))
        {
            // case 1: cache contains all data
            memcpy(read_buf, cache_buf + cur_off, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // case 2: cache contains some data
            uint64_t cached_bytes = cache_size - cur_off;
            if (n_bytes - cached_bytes > fsize - reader.tellg())
            {
                std::stringstream stream;
                stream << "Reading beyond end of file" << std::endl;
                stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes << " fsize: " << fsize
                       << " current pos:" << reader.tellg() << std::endl;
                throw std::runtime_error(stream.str());
            }
            memcpy(read_buf, cache_buf + cur_off, cached_bytes);

            // go to disk and fetch more data
            reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
            // reset cur off
            cur_off = cache_size;

            uint64_t size_left = fsize - reader.tellg();

            if (size_left >= cache_size)
            {
                reader.read(cache_buf, cache_size);
                cur_off = 0;
            }
            // note that if size_left < cache_size, then cur_off = cache_size,
            // so subsequent reads will all be directly from file
        }
    }

  private:
    // underlying ifstream
    std::ifstream reader;
    // # bytes to cache in one shot read
    uint64_t cache_size = 0;
    // underlying buf for cache
    char *cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;
    // file size
    uint64_t fsize = 0;
};

// sequential cached writes
class cached_ofstream
{
  public:
    cached_ofstream(const std::string &filename, uint64_t cache_size) : cache_size(cache_size), cur_off(0)
    {
        writer.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            writer.open(filename, std::ios::binary);
            if (cache_size == 0) {
                throw std::runtime_error("cache_size is 0");
            }

            if (!writer.is_open()) {
                throw std::io_errc();
            }
            cache_buf = new char[cache_size];
            LOG_INFO("Opened: %s, cache_size: %lu", filename.c_str(), cache_size);
        }
        catch (std::system_error &e)
        {
            LOG_ERROR("io_error");
            throw e;
        }
    }

    ~cached_ofstream()
    {
        this->close();
    }

    void close()
    {
        // dump any remaining data in memory
        if (cur_off > 0)
        {
            this->flush_cache();
        }

        if (cache_buf != nullptr)
        {
            delete[] cache_buf;
            cache_buf = nullptr;
        }

        if (writer.is_open())
            writer.close();
        LOG_INFO("Finished writing %luB", fsize);
    }

    size_t get_file_size()
    {
        return fsize;
    }
    // writes n_bytes from write_buf to the underlying ofstream/cache
    void write(char *write_buf, uint64_t n_bytes)
    {
        if (cache_buf == nullptr) {
            throw std::runtime_error("cache_buf is null");
        }
        if (n_bytes <= (cache_size - cur_off))
        {
            // case 1: cache can take all data
            memcpy(cache_buf + cur_off, write_buf, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // case 2: cache cant take all data
            // go to disk and write existing cache data
            writer.write(cache_buf, cur_off);
            fsize += cur_off;
            // write the new data to disk
            writer.write(write_buf, n_bytes);
            fsize += n_bytes;
            // memset all cache data and reset cur_off
            memset(cache_buf, 0, cache_size);
            cur_off = 0;
        }
    }

    void flush_cache()
    {
        if (cache_buf == nullptr) {
            throw std::runtime_error("cache_buf is null");
        }
        writer.write(cache_buf, cur_off);
        fsize += cur_off;
        memset(cache_buf, 0, cache_size);
        cur_off = 0;
    }

    void reset()
    {
        flush_cache();
        writer.seekp(0);
    }

  private:
    // underlying ofstream
    std::ofstream writer;
    // # bytes to cache for one shot write
    uint64_t cache_size = 0;
    // underlying buf for cache
    char *cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;

    // file size
    uint64_t fsize = 0;
};


MERCURY_NAMESPACE_END(core);