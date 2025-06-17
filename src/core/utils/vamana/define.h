#pragma once

#include <sstream>
#include "src/core/common/common.h"

MERCURY_NAMESPACE_BEGIN(core);

// setting it to 8 because that works well for AVX2. If we have AVX512
// implementations of distance algos, they might have to set this to 16
#define DEFAULT_ALIGNMENT_FACTOR 8

#define DEFAULT_MAXC 750

#define MAX_POINTS_FOR_USING_BITSET 10000000

#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)

MERCURY_NAMESPACE_END(core);