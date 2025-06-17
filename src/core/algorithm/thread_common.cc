#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "thread_common.h"
#include "src/core/utils/string_util.h"



MERCURY_NAMESPACE_BEGIN(core);

bool GetEnvBool(const char* key, bool default_value) {
    char *value = getenv(key);
    if (value == NULL) {
        return default_value;
    }

    if (strcmp(value, "True") == 0 || strcmp(value, "true") == 0) {
        return true;
    }

    if (strcmp(value, "false") == 0 || strcmp(value, "False") == 0) {
        return false;
    }

    return default_value;
}

int GetEnvInt(const char* key, int default_value) {
    char *value = getenv(key);
    if (value == NULL) {
        return default_value;
    }

    int int_value = 0;
    if (StringUtil::strToInt32(value, int_value)) {
        return int_value;
    }

    return default_value;
}

bool mercury_need_parallel = GetEnvBool("mercury_need_parallel", false);
int mercury_concurrency = GetEnvInt("mercury_concurrency", DEFAULT_MERCURY_SEARCH_CONCURRENCY);
int mercury_no_concurrency_count = GetEnvInt("mercury_no_concurrency_count", DEFAULT_MERCURY_NO_CONCURRENT_COUNT);
int mercury_pool_size = GetEnvInt("mercury_pool_size", DEFAULT_MERCURY_THREAD);
int mercury_min_parralel_centroids = GetEnvInt("mercury_min_parralel_centroids", DEFAULT_MIN_PARRALEL_CENTROIDS);
int mercury_doc_num_per_concurrency = GetEnvInt("mercury_doc_num_per_concurrency", DEFAULT_MERCURY_DOC_NUM_PER_CONCURRENCY);
int mercury_max_concurrency_num = GetEnvInt("mercury_max_concurrency_num", DEFAULT_MERCURY_MAX_CONCURRENCY_NUM);

void SetThreadEnv() {
    mercury_need_parallel = GetEnvBool("mercury_need_parallel", false);
    mercury_concurrency = GetEnvInt("mercury_concurrency", DEFAULT_MERCURY_SEARCH_CONCURRENCY);
    mercury_no_concurrency_count = GetEnvInt("mercury_no_concurrency_count", DEFAULT_MERCURY_NO_CONCURRENT_COUNT);
    mercury_pool_size = GetEnvInt("mercury_pool_size", DEFAULT_MERCURY_THREAD);
    mercury_min_parralel_centroids = GetEnvInt("mercury_min_parralel_centroids", DEFAULT_MIN_PARRALEL_CENTROIDS);
    mercury_doc_num_per_concurrency = GetEnvInt("mercury_doc_num_per_concurrency", DEFAULT_MERCURY_DOC_NUM_PER_CONCURRENCY);
    mercury_max_concurrency_num = GetEnvInt("mercury_max_concurrency_num", DEFAULT_MERCURY_MAX_CONCURRENCY_NUM);
}

MERCURY_NAMESPACE_END(core);
