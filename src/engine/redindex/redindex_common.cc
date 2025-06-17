/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-25 15:37

#include "redindex_common.h"
namespace mercury {
namespace redindex {
using namespace mercury::core;

bool SchemaToIndexParam(const SchemaParams &schema, IndexParams &index_params)
{
    if (schema.find(IvfPQScanNum) != schema.end()) {
        index_params.set(PARAM_PQ_SCAN_NUM, std::stoul(schema.at(IvfPQScanNum)));
    }

    if (schema.find(IvfCoarseScanRatio) != schema.end()) {
        index_params.set(PARAM_COARSE_SCAN_RATIO, std::stof(schema.at(IvfCoarseScanRatio)));
    }

    if (schema.find(PARAM_FINE_SCAN_RATIO) != schema.end()) {
        index_params.set(PARAM_FINE_SCAN_RATIO, std::stof(schema.at(PARAM_FINE_SCAN_RATIO)));
    }

    if (schema.find(PARAM_RT_COARSE_SCAN_RATIO) != schema.end()) {
        index_params.set(PARAM_RT_COARSE_SCAN_RATIO, std::stof(schema.at(PARAM_RT_COARSE_SCAN_RATIO)));
    }

    if (schema.find(PARAM_RT_FINE_SCAN_RATIO) != schema.end()) {
        index_params.set(PARAM_RT_FINE_SCAN_RATIO, std::stof(schema.at(PARAM_RT_FINE_SCAN_RATIO)));
    }

    if (schema.find(TrainDataPath) != schema.end()) {
        index_params.set(PARAM_TRAIN_DATA_PATH, schema.at(TrainDataPath));
    }

    if (schema.find(DataType) != schema.end()) {
        index_params.set(PARAM_DATA_TYPE, schema.at(DataType));
    }

    if (schema.find(Method) != schema.end()) {
        index_params.set(PARAM_METHOD, schema.at(Method));
    }

    if (schema.find(Dimension) != schema.end()) {
        index_params.set(PARAM_DIMENSION, std::stoul(schema.at(Dimension)));
    }

    if (schema.find(PqFragmentCnt) != schema.end()) {
        index_params.set(PARAM_PQ_FRAGMENT_NUM, std::stoul(schema.at(PqFragmentCnt)));
    }

    if (schema.find(PqCentroidNum) != schema.end()) {
        index_params.set(PARAM_PQ_CENTROID_NUM, std::stoul(schema.at(PqCentroidNum)));
    }

    if (schema.find(GroupIvfVisitLimit) != schema.end()) {
        index_params.set(PARAM_GROUP_IVF_VISIT_LIMIT, std::stoul(schema.at(GroupIvfVisitLimit)));
    }

    if (schema.find(PARAM_GROUP_IVF_VISIT_LIMIT) != schema.end()) {
        index_params.set(PARAM_GROUP_IVF_VISIT_LIMIT, std::stoul(schema.at(PARAM_GROUP_IVF_VISIT_LIMIT)));
    }

    if (schema.find(PARAM_GENERAL_RESERVED_DOC) != schema.end()) {
        index_params.set(PARAM_GENERAL_RESERVED_DOC, std::stoul(schema.at(PARAM_GENERAL_RESERVED_DOC)));
    }

    if (schema.find(PARAM_GENERAL_MAX_BUILD_NUM) != schema.end()) {
        index_params.set(PARAM_GENERAL_MAX_BUILD_NUM, std::stoul(schema.at(PARAM_GENERAL_MAX_BUILD_NUM)));
    }

    if (schema.find(PARAM_SORT_BUILD_GROUP_LEVEL) != schema.end()) {
        index_params.set(PARAM_SORT_BUILD_GROUP_LEVEL, std::stoul(schema.at(PARAM_SORT_BUILD_GROUP_LEVEL)));
    }

    if (schema.find(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE) != schema.end()) {
        const std::string &contain = schema.at(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_GENERAL_CONTAIN_FEATURE_PROFILE, true);
        }
    }

    if (schema.find(IndexType) != schema.end()) {
        const std::string &schema_type = schema.at(IndexType);
        if (schema_type == "IvfFlat") {
            index_params.set(PARAM_INDEX_TYPE, "Ivf");
        } else if (schema_type == "IvfPQFlat") {
            index_params.set(PARAM_INDEX_TYPE, "IvfPQ");
        } else if (schema_type == "GroupIvfFlat") {
            index_params.set(PARAM_INDEX_TYPE, "GroupIvf");
        } else {
            index_params.set(PARAM_INDEX_TYPE, schema_type);
        }
    }

    if (schema.find(PARAM_CUSTOMED_PART_DIMENSION) != schema.end()) {
        index_params.set(PARAM_CUSTOMED_PART_DIMENSION, std::stoul(schema.at(PARAM_CUSTOMED_PART_DIMENSION)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE, std::stoul(schema.at(PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST, std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_ALPHA) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_ALPHA, std::stof(schema.at(PARAM_VAMANA_INDEX_BUILD_ALPHA)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED) != schema.end()) {
        const std::string &saturate = schema.at(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED);
        if (saturate == "True" || saturate == "true") {
            index_params.set(PARAM_VAMANA_INDEX_BUILD_IS_SATURATED, true);
        }
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION, std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM, std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_THREAD_NUM)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT, std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION) != schema.end()) {
        const std::string &partition = schema.at(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION);
        if (partition == "True" || partition == "true") {
            index_params.set(PARAM_VAMANA_INDEX_BUILD_IS_PARTITION, true);
        }
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM,
                        std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR,
                        std::stoul(schema.at(PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_SEARCH_L) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_SEARCH_L,
                        std::stoul(schema.at(PARAM_VAMANA_INDEX_SEARCH_L)));
    }

    if (schema.find(PARAM_VAMANA_INDEX_SEARCH_BW) != schema.end()) {
        index_params.set(PARAM_VAMANA_INDEX_SEARCH_BW,
                        std::stoul(schema.at(PARAM_VAMANA_INDEX_SEARCH_BW)));
    }

    if (schema.find(PARAM_DISKANN_BUILD_MODE) != schema.end()) {
        index_params.set(PARAM_DISKANN_BUILD_MODE, schema.at(PARAM_DISKANN_BUILD_MODE));
    }

    if (schema.find(PARAM_GROUP_HNSW_BUILD_THRESHOLD) != schema.end()) {
        index_params.set(PARAM_GROUP_HNSW_BUILD_THRESHOLD, std::stoul(schema.at(PARAM_GROUP_HNSW_BUILD_THRESHOLD)));
    }

    if (schema.find(PARAM_HNSW_BUILDER_MAX_LEVEL) != schema.end()) {
        index_params.set(PARAM_HNSW_BUILDER_MAX_LEVEL, std::stoul(schema.at(PARAM_HNSW_BUILDER_MAX_LEVEL)));
    }

    if (schema.find(PARAM_HNSW_BUILDER_SCALING_FACTOR) != schema.end()) {
        index_params.set(PARAM_HNSW_BUILDER_SCALING_FACTOR, std::stoul(schema.at(PARAM_HNSW_BUILDER_SCALING_FACTOR)));
    }

    if (schema.find(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT) != schema.end()) {
        index_params.set(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT,
                         std::stoul(schema.at(PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT)));
    }

    if (schema.find(PARAM_GRAPH_COMMON_SEARCH_STEP) != schema.end()) {
        index_params.set(PARAM_GRAPH_COMMON_SEARCH_STEP, std::stoul(schema.at(PARAM_GRAPH_COMMON_SEARCH_STEP)));
    }

    if (schema.find(PARAM_HNSW_BUILDER_EFCONSTRUCTION) != schema.end()) {
        index_params.set(PARAM_HNSW_BUILDER_EFCONSTRUCTION, std::stoul(schema.at(PARAM_HNSW_BUILDER_EFCONSTRUCTION)));
    }

    if (schema.find(PARAM_GRAPH_COMMON_MAX_SCAN_NUM) != schema.end()) {
        index_params.set(PARAM_GRAPH_COMMON_MAX_SCAN_NUM, std::stoul(schema.at(PARAM_GRAPH_COMMON_MAX_SCAN_NUM)));
    }

    if (schema.find(PARAM_CUSTOM_DISTANCE_METHOD) != schema.end()) {
        index_params.set(PARAM_CUSTOM_DISTANCE_METHOD, schema.at(PARAM_CUSTOM_DISTANCE_METHOD));
    }

    if (schema.find(PARAM_VECTOR_ENABLE_GPU) != schema.end()) {
        const std::string &contain = schema.at(PARAM_VECTOR_ENABLE_GPU);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_VECTOR_ENABLE_GPU, true);
        }
    }

    if (schema.find(PARAM_VECTOR_ENABLE_GPU_INMEM) != schema.end()) {
        const std::string &contain = schema.at(PARAM_VECTOR_ENABLE_GPU_INMEM);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_VECTOR_ENABLE_GPU_INMEM, true);
        }
    }

    if (schema.find(PARAM_VECTOR_ENABLE_GPU_PLUS) != schema.end()) {
        const std::string &contain = schema.at(PARAM_VECTOR_ENABLE_GPU_PLUS);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_VECTOR_ENABLE_GPU_PLUS, true);
        }
    }

    if (schema.find(PARAM_VECTOR_ENABLE_BATCH) != schema.end()) {
        const std::string &contain = schema.at(PARAM_VECTOR_ENABLE_BATCH);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_VECTOR_ENABLE_BATCH, true);
        }
    }

    if (schema.find(PARAM_VECTOR_ENABLE_HALF) != schema.end()) {
        const std::string &contain = schema.at(PARAM_VECTOR_ENABLE_HALF);
        if (contain == "True" || contain == "true") {
            index_params.set(PARAM_VECTOR_ENABLE_HALF, true);
        }
    }

    if (schema.find(PARAM_TABLE_NAME) != schema.end()) {
        index_params.set(PARAM_VECTOR_INDEX_NAME, schema.at(PARAM_TABLE_NAME));
    }

    if (schema.find(PARAM_VECTOR_ENABLE_DEVICE_NO) != schema.end()) {
        index_params.set(PARAM_VECTOR_ENABLE_DEVICE_NO, std::stoul(schema.at(PARAM_VECTOR_ENABLE_DEVICE_NO)));
    }

    if (schema.find(PARAM_MULTI_AGE_MODE) != schema.end()) {
        const std::string &multi_age_mode = schema.at(PARAM_MULTI_AGE_MODE);
        if (multi_age_mode == "True" || multi_age_mode == "true") {
            index_params.set(PARAM_MULTI_AGE_MODE, true);
        }
    }

    if (schema.find(PARAM_ENABLE_MIPS) != schema.end()) {
        const std::string &enable_mips = schema.at(PARAM_ENABLE_MIPS);
        if (enable_mips == "True" || enable_mips == "true") {
            index_params.set(PARAM_ENABLE_MIPS, true);
        }
    }

    if (schema.find(PARAM_ENABLE_FORCE_HALF) != schema.end()) {
        const std::string &force_half = schema.at(PARAM_ENABLE_FORCE_HALF);
        if (force_half == "True" || force_half == "true") {
            index_params.set(PARAM_ENABLE_FORCE_HALF, true);
        }
    }

    if (schema.find(PARAM_DELMAP_BEFORE) != schema.end()) {
        const std::string &delmap_before = schema.at(PARAM_DELMAP_BEFORE);
        if (delmap_before == "True" || delmap_before == "true") {
            index_params.set(PARAM_DELMAP_BEFORE, true);
        }
    }

    if (schema.find(PARAM_ENABLE_FINE_CLUSTER) != schema.end()) {
        const std::string &enable_fine_cluster = schema.at(PARAM_ENABLE_FINE_CLUSTER);
        if (enable_fine_cluster == "True" || enable_fine_cluster == "true") {
            index_params.set(PARAM_ENABLE_FINE_CLUSTER, true);
        }
    }

    if (schema.find(PARAM_ENABLE_RESIDUAL) != schema.end()) {
        const std::string &enable_residual = schema.at(PARAM_ENABLE_RESIDUAL);
        if (enable_residual == "True" || enable_residual == "true") {
            index_params.set(PARAM_ENABLE_RESIDUAL, true);
        }
    }

    if (schema.find(PARAM_ENABLE_QUANTIZE) != schema.end()) {
        const std::string &enable_quantize = schema.at(PARAM_ENABLE_QUANTIZE);
        if (enable_quantize == "True" || enable_quantize == "true") {
            index_params.set(PARAM_ENABLE_QUANTIZE, true);
        }
    }

    if (schema.find(PARAM_MONOCEROS_ROOT_DIR) != schema.end()) {
        index_params.set(PARAM_MONOCEROS_ROOT_DIR, schema.at(PARAM_MONOCEROS_ROOT_DIR));
    }

    if (schema.find(PARAM_MONOCEROS_IS_ONLINE) != schema.end()) {
        const std::string &is_online = schema.at(PARAM_MONOCEROS_IS_ONLINE);
        if (is_online == "True" || is_online == "true") {
            index_params.set(PARAM_MONOCEROS_IS_ONLINE, true);
        }
    }

    return true;
}

} // namespace redindex
} // namespace mercury
