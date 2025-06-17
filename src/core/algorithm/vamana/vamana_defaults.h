#pragma once

#include <stdint.h>

#include "src/core/framework/index_logger.h"
#include "src/core/common/common.h"


MERCURY_NAMESPACE_BEGIN(core);
namespace defaults
{
const float ALPHA = 1.2f;
const uint32_t NUM_THREADS = 0;
const uint32_t MAX_OCCLUSION_SIZE = 750;
const bool HAS_LABELS = false;
const uint32_t FILTER_LIST_SIZE = 0;
const uint32_t NUM_FROZEN_POINTS_STATIC = 0;
const uint32_t NUM_FROZEN_POINTS_DYNAMIC = 1;

// In-mem index related limits
const float GRAPH_SLACK_FACTOR = 1.3;

// SSD Index related limits
const uint64_t MAX_GRAPH_DEGREE = 512;
const uint64_t SECTOR_LEN = 4096;
const uint64_t MAX_N_SECTOR_READS = 128;

// Search AIO Context Num
const uint32_t IO_CONTEXT_NUM = 64;

// Search AIO MAX NR (256k)
const uint32_t AIO_MAX_NR = 524288;

// CMD SET AIO MAX NR
const char* const CMD_SET_AIO_MAX_NR = "echo fs.aio-max-nr=524288 | sudo tee /etc/sysctl.conf && sudo sysctl -p";

// CMD GET AIO MAX NR
const char* const CMD_GET_AIO_MAX_NR = "sudo cat /proc/sys/fs/aio-max-nr";

// Search Cache Ratio (3%)
const float CACHE_RATIO = 0.03f;
const size_t CACHE_LOAD_BLOCK_SIZE = 4096;
const size_t CACHE_LEVEL_BLOCK_SIZE = 2048;

// Search Beam Width
const uint32_t SEARCH_BEAM_WIDTH = 4;

// Vamana Index
const uint32_t MAX_DEGREE = 64;
const uint32_t BUILD_LIST_SIZE = 100;
const uint32_t SATURATE_GRAPH = false;
const uint32_t SEARCH_LIST_SIZE = 140;
const uint32_t MAX_RAM_SEARCH_LIST_SIZE = 500;
const uint32_t MAX_RAM_SEARCH_THREAD_NUM = 64;

// Partition Build
const uint32_t K_BASE = 2;

// Common
const uint32_t MEM_ALLOC_EXPAND_FACTOR = 2;
} // namespace defaults


MERCURY_NAMESPACE_END(core);

