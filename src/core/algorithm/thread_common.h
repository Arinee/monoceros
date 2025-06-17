/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <qiuming@xiaohongshu.com>
/// Created: 2019-08-29 15:37

#pragma once

#include "src/core/common/common.h"
#include "src/core/framework/utility/thread_pool.h"

MERCURY_NAMESPACE_BEGIN(core);

//=============server config====================
constexpr auto DEFAULT_MERCURY_SEARCH_CONCURRENCY(5);
constexpr auto DEFAULT_MERCURY_NO_CONCURRENT_COUNT(4);
constexpr auto DEFAULT_MERCURY_THREAD(30);
constexpr auto DEFAULT_MIN_PARRALEL_CENTROIDS(500);
constexpr auto DEFAULT_MERCURY_DOC_NUM_PER_CONCURRENCY(10000);
constexpr auto DEFAULT_MERCURY_MAX_CONCURRENCY_NUM(15);
//=============end server config=================

extern bool mercury_need_parallel;
extern int mercury_concurrency;
extern int mercury_no_concurrency_count;
extern int mercury_pool_size;
extern int mercury_min_parralel_centroids;
extern int mercury_doc_num_per_concurrency;
extern int mercury_max_concurrency_num;

static std::unique_ptr<mercury::ThreadPool> _pool;

void SetThreadEnv();

MERCURY_NAMESPACE_END(core);
