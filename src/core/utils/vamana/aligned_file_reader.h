#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <libaio.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#include <iostream>
#include "src/core/common/common.h"
#include "src/core/framework/index_framework.h"

#define MAX_EVENTS 1024

typedef io_context_t IOContext;
typedef int FileHandle;

MERCURY_NAMESPACE_BEGIN(core);

struct AlignedRead
{
    uint64_t offset; // where to read from
    uint64_t len;    // how much to read
    void *buf;       // where to read into

    AlignedRead() : offset(0), len(0), buf(nullptr)
    {
    }

    AlignedRead(uint64_t offset, uint64_t len, void *buf) : offset(offset), len(len), buf(buf)
    {
        // assert(IS_512_ALIGNED(offset));
        // assert(IS_512_ALIGNED(len));
        // assert(IS_512_ALIGNED(buf));
        // assert(malloc_usable_size(buf) >= len);
    }
};

class AlignedFileReader
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;
    io_context_t bad_ctx = (io_context_t)-1;
    tsl::robin_map<int32_t, IOContext> ctx_map;
    std::mutex ctx_mut;

  public:
    AlignedFileReader();
    ~AlignedFileReader();

    IOContext &get_ctx(int32_t id);

    // register thread-id for a context
    void register_thread(int32_t id);

    // de-register thread-id for a context
    void deregister_thread(int32_t id);
    void deregister_all_threads();

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname);
    void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false);
};

MERCURY_NAMESPACE_END(core);