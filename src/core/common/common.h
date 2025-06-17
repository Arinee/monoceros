/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: qiuming <qiuming@xiaohongshu.com>
/// Created: 2019-08-29 15:37

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <cstring>
#include <random>

namespace mercury { namespace core {
using SlotIndex = uint32_t;
using docid_t = uint32_t;
using exdocid_t = int64_t;
using pk_t = uint64_t;
using segid_t = uint32_t;
//! high 32 is segmentid, low 32 is docid
using gloid_t = uint64_t;
using idx_t = uint32_t;
using key_t = uint64_t;
using cat_t = std::uint64_t;
using slot_t = std::int32_t;
using level_t = std::uint32_t;
using group_t = std::uint32_t;
using gindex_t = std::uint32_t;

constexpr SlotIndex INVALID_SLOT_INDEX = std::numeric_limits<SlotIndex>::max();
constexpr pk_t INVALID_PK = std::numeric_limits<pk_t>::max();

using DocIdMap = std::unordered_map<uint32_t, uint32_t>;
constexpr docid_t INVALID_DOC_ID = std::numeric_limits<docid_t>::max();
const uint64_t INVALID_OFFSET = std::numeric_limits<uint64_t>::max();
const key_t INVALID_KEY = std::numeric_limits<key_t>::max();
const docid_t INVALID_SEGID = std::numeric_limits<segid_t>::max();
const gloid_t INVALID_GLOID = std::numeric_limits<gloid_t>::max();
const gindex_t INVALID_GROUP_INDEX = std::numeric_limits<gindex_t>::max();
const int SIZE_WRITE_ONE_TIME = 10 * 1024 * 1024;
const double BLOCK_USE_RATE = 0.5;
const auto INVALID_CAT_ID = std::numeric_limits<cat_t>::max();


constexpr auto COMPONENT_FEATURE_META("feature_meta");
constexpr auto COMPONENT_COARSE_INDEX("coarse_index.dat");
constexpr auto COMPONENT_FASTSCAN_INDEX("fastscan_index.dat");
constexpr auto COMPONENT_COARSE_HNSW_INDEX("coarse_hnsw_index.dat");
constexpr auto COMPONENT_ROUGH_MATRIX("rough_matrix");
constexpr auto COMPONENT_FINE_ROUGH_MATRIX("fine_rough_matrix");
constexpr auto COMPONENT_INTEGRATE_MATRIX("integrate_matrix");
constexpr auto COMPONENT_PQ_CODE_PROFILE("pq_code_profile");
constexpr auto COMPONENT_RANK_SCORE_PROFILE("rank_score_profile");
constexpr auto COMPONENT_SLOT_INDEX_PROFILE("slot_index_profile");
constexpr auto COMPONENT_REDINDEX_DOCID_ARRAY("redindex_docid_array"); //为了兼容
constexpr auto COMPONENT_GROUP_MANAGER("group_manager");
constexpr auto COMPONENT_GROUP_HNSW_INFO_MANAGER("group_hnsw_info_manager");
constexpr auto COMPONENT_MAX_DOC_NUM("group_max_doc_num");
constexpr auto COMPONENT_DOC_NUM("group_doc_num");
constexpr auto COMPONENT_SORT_BUILD_GROUP_LEVEL("sort_build_group_level");
constexpr auto COMPONENT_BUILD_THRESHOLD("build_threshold");
constexpr auto COMPONENT_MAXLEVEL("maxLevel");
constexpr auto COMPONENT_SCALING_FACTOR("scalingFactor");
constexpr auto COMPONENT_UPPER_NEIGHBOR_CNT("upperNeighborCnt");
constexpr auto COMPONENT_EF_CONSTRUCTION("efConstruction");
constexpr auto COMPONENT_VAMANA_NEIGHBOR_R("vamana_neighbor_r");
constexpr auto COMPONENT_VAMANA_CADIDATE_L("vamana_cadidate_l");

// =========== for SQ8 =================
constexpr auto COMPONENT_VMAX("vmax");
constexpr auto COMPONENT_VMIN("vmin");

constexpr auto COMPONENT_PK_PROFILE("pk_profile.dat");
constexpr auto COMPONENT_PRODUCT_PROFILE("product_profile.dat");
constexpr auto COMPONENT_CENT_REFINE_PROFILE("cent_refine_profile.dat");
constexpr auto COMPONENT_FEATURE_PROFILE("feature_profile.dat");
constexpr auto COMPONENT_DOC_CREATE_TIME_PROFILE("doc_create_time_profile.dat");
constexpr auto COMPONENT_MIPS_NORM("mips_norm.dat");
constexpr auto COMPONENT_IDMAP("idmap.dat");
constexpr auto COMPONENT_DELETEMAP("delete_map.dat");

// ================index params name==================
constexpr auto PARAM_GENERAL_INDEX_MEMORY_QUOTA("mercury.general.index.memory_quota");
constexpr auto PARAM_GENERAL_MAX_BUILD_NUM("mercury.general.index.max_build_num");
constexpr auto PARAM_GENERAL_RECALL_TEST_MODE("mercury.general.recall_test_mode");
constexpr auto PARAM_GENERAL_RESERVED_DOC("mercury.general.reserved_doc_num");
constexpr auto PARAM_GENERAL_MAX_GROUP_LEVEL_NUM("mercury.general.index.max_group_level_num");
constexpr auto PARAM_GENERAL_CONTAIN_FEATURE_PROFILE("mercury.general.index.contain_feature_profile");
constexpr auto PARAM_CUSTOMED_PART_DIMENSION("mercury.general.index.part_dimension");
constexpr auto PARAM_COARSE_SCAN_RATIO("mercury.coarse_scan_ratio");
constexpr auto PARAM_FINE_SCAN_RATIO("mercury.fine_scan_ratio");
constexpr auto PARAM_RT_COARSE_SCAN_RATIO("mercury.rt_coarse_scan_ratio");
constexpr auto PARAM_RT_FINE_SCAN_RATIO("mercury.rt_fine_scan_ratio");
constexpr auto PARAM_DOWNGRADE_PERCENT("mercury.downgrade_percent");
constexpr auto PARAM_PQ_SCAN_NUM("mercury.pq_scan_num"); //改造链路让上游自己在查询串中指定
constexpr auto PARAM_BIAS_VECTOR("mercury.bias_vector");
constexpr auto PARAM_TRAIN_DATA_PATH("mercury.general.train_data_path");
constexpr auto PARAM_DATA_TYPE("mercury.general.data_type");
constexpr auto PARAM_METHOD("mercury.general.method");
constexpr auto PARAM_DIMENSION("mercury.general.dimension");
constexpr auto PARAM_PQ_FRAGMENT_NUM("mercury.pq.fragment_num");
constexpr auto PARAM_PQ_CENTROID_NUM("mercury.pq.centroid_num");
constexpr auto PARAM_INDEX_TYPE("mercury.general.index_type");
constexpr auto PARAM_WITH_PK("mercury.general.index.with_pk");
constexpr auto PARAM_GROUP_IVF_VISIT_LIMIT("mercury.group_ivf.visit_limit");
constexpr auto PARAM_GROUP_IVF_VISIT_LIMIT_GROUP0("mercury.group_ivf.visit_limit_group_0");
constexpr auto PARAM_CUSTOM_DISTANCE_METHOD("mercury.general.custom_distance_method");
constexpr auto PARAM_MULTI_QUERY_MODE("mercury.general.multi_query_mode");
constexpr auto PARAM_MULTI_AGE_MODE("mercury.general.multi_age_mode");
constexpr auto PARAM_SORT_BUILD_GROUP_LEVEL("mercury.sort.build.group.level");
constexpr auto PARAM_VECTOR_ENABLE_BATCH("mercury.general.vector_enable_batch");
constexpr auto PARAM_VECTOR_ENABLE_GPU("mercury.general.vector_enable_gpu");
constexpr auto PARAM_VECTOR_ENABLE_GPU_INMEM("mercury.general.vector_enable_gpu_in_mem");
constexpr auto PARAM_VECTOR_ENABLE_GPU_PLUS("mercury.general.vector_enable_gpu_plus");
constexpr auto PARAM_VECTOR_INDEX_NAME("mercury.general.vector_index_name");
constexpr auto PARAM_VECTOR_ENABLE_HALF("mercury.general.vector_enable_half");
constexpr auto PARAM_VECTOR_ENABLE_DEVICE_NO("mercury.general.vector_enable_device_no");
constexpr auto PARAM_TABLE_NAME("table_name");
constexpr auto PARAM_ENABLE_MIPS("mercury.general.enable_mips");
constexpr auto PARAM_DELMAP_BEFORE("mercury.general.delmap_before");
constexpr auto PARAM_ENABLE_FORCE_HALF("mercury.general.enable_force_half");
constexpr auto PARAM_ENABLE_FINE_CLUSTER("mercury.general.enable_fine_cluster");
constexpr auto PARAM_ENABLE_RESIDUAL("mercury.general.enable_residual");
constexpr auto PARAM_ENABLE_QUANTIZE("mercury.general.enable_quantize");
constexpr auto PARAM_MONOCEROS_IS_ONLINE("monoceros.general.is_online");
constexpr auto PARAM_MONOCEROS_ROOT_DIR("monoceros.general.root_dir");

//HNSW
constexpr auto PARAM_GROUP_HNSW_BUILD_THRESHOLD("mercury.group_hnsw.build_threshold");
constexpr auto PARAM_HNSW_BUILDER_MAX_LEVEL("mercury.hnsw.builder.max_level");
constexpr auto PARAM_HNSW_BUILDER_SCALING_FACTOR("mercury.hnsw.builder.scaling_factor");
constexpr auto PARAM_HNSW_BUILDER_UPPER_NEIGHBOR_CNT("mercury.hnsw.builder.upper_neighbor_cnt");
constexpr auto PARAM_GRAPH_COMMON_SEARCH_STEP("mercury.graph.common.search_step");
constexpr auto PARAM_HNSW_BUILDER_EFCONSTRUCTION("mercury.hnsw.builder.efconstruction");
constexpr auto PARAM_GRAPH_COMMON_MAX_SCAN_NUM("mercury.graph.common.max_scan_num");
constexpr auto PARAM_GRAPH_MAX_SCAN_NUM_IN_QUERY("mercury.graph.max_scan_num_in_query");

constexpr auto PARAM_GRAPH_COMMON_MAX_DOC_CNT("mercury.graph.common.max_doc_cnt");
constexpr auto PARAM_GRAPH_COMMON_COMBO_FILE("mercury.graph.common.combo_file");
constexpr auto PARAM_GRAPH_COMMON_NEIGHBOR_CNT("mercury.graph.common.neighbor_cnt");
constexpr auto PARAM_GRAPH_COMMON_GRAPH_TYPE("mercury.graph.common.graph_type");
constexpr auto PARAM_GRAPH_BUILDER_HOLD_DISTANCE_IN_BUILDING("mercury.graph.builder.hold_distance_in_building");
constexpr auto PARAM_GRAPH_BUILDER_MEMORY_QUOTA("mercury.graph.builder.memory_quota");
constexpr auto PARAM_GRAPH_BUILD_MODE("mercury.graph.builder.build_mode");
constexpr auto PARAM_GRAPH_BUILDER_WORK_QUEUE_SIZE("mercury.graph.builder.work_queue_size");
constexpr auto PARAM_HNSW_COMMON_SEARCH_NEIGHBOR_METHOD("mercury.hnsw.common.search_neighbor_method");
constexpr auto PARAM_HNSW_SEARCHER_EF("mercury.hnsw.searcher.ef");

//VAMANA
constexpr auto PARAM_VAMANA_INDEX_MAX_GRAPH_DEGREE("mercury.vamana.index.max_graph_degree");
constexpr auto PARAM_VAMANA_INDEX_BUILD_MAX_SEARCH_LIST("mercury.vamana.index.build.max_search_list");
constexpr auto PARAM_VAMANA_INDEX_BUILD_ALPHA("mercury.vamana.index.build.alpha");
constexpr auto PARAM_VAMANA_INDEX_BUILD_IS_SATURATED("mercury.vamana.index.build.is_saturated");
constexpr auto PARAM_VAMANA_INDEX_BUILD_MAX_OCCLUSION("mercury.vamana.index.build.max_occlusion");
constexpr auto PARAM_VAMANA_INDEX_BUILD_THREAD_NUM("mercury.vamana.index.build.thread_num");
constexpr auto PARAM_VAMANA_INDEX_BUILD_BATCH_COUNT("mercury.vamana.index.build.batch_count");
constexpr auto PARAM_VAMANA_INDEX_BUILD_IS_PARTITION("mercury.vamana.index.build.is_partition");
constexpr auto PARAM_VAMANA_INDEX_BUILD_MAX_SHARD_DATA_NUM("mercury.vamana.index.build.max_shard_data_num");
constexpr auto PARAM_VAMANA_INDEX_BUILD_DUPLICATE_FACTOR("mercury.vamana.index.build.duplicate_factor");
constexpr auto PARAM_DISKANN_BUILD_MODE("mercury.diskann.build.mode");
constexpr auto PARAM_VAMANA_INDEX_SEARCH_L("mercury.vamana.index.search.l");
constexpr auto PARAM_VAMANA_INDEX_SEARCH_BW("mercury.vamana.index.search.bw");
// ================end index params name==================

// ===============index name=========================
constexpr auto Ivf("Ivf"); //TODO config change
constexpr auto IvfFlat("IvfFlat");
constexpr auto IvfPQ("IvfPQ");
constexpr auto GroupIvf("GroupIvf"); //TODO config change
constexpr auto GroupHnsw("GroupHnsw");
constexpr auto GroupIvfPq("GroupIvfPq");
constexpr auto GroupIvfPqSmall("GroupIvfPqSmall");
constexpr auto GpuGroupIvf("GpuGroupIvf");
constexpr auto DiskVamana("DiskVamana");
constexpr auto RamVamana("RamVamana");
constexpr auto IvfFastScan("IvfFastScan");
constexpr auto IvfRpq("IvfRpq");
//===============end index name=====================

constexpr auto INDEX_META_FILE("/index.meta");
constexpr auto MIPS_META_FILE("/mips.meta");
constexpr auto IVF_CENTROID_FILE_POSTFIX("/coarse/coarse.dat");
constexpr auto IVF_FINE_CENTROID_FILE_POSTFIX("/fine/fine.dat");
constexpr auto PQ_CENTROID_FILE_MIDDLEFIX("/integrate/integrate_");
constexpr auto PQ_CENTROID_FILE_POSTFIX(".dat");
constexpr auto DefaultLevelCnt(1);
constexpr auto DefaultPqFragmentCnt(32);
constexpr auto DefaultPqCentroidNum(256);
constexpr auto DEFAULT_RESERVED_DOC(10000);
constexpr auto DEFAULT_MAX_GROUP_LEVEL_NUM(6);

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define MERCURY_NAMESPACE_BEGIN(x) namespace mercury { namespace x {
#define MERCURY_NAMESPACE_END(x) }}

bool inline isAligned32(const void* ptr) {
    return ((uintptr_t)(ptr) & 0x1F) == 0;
}

}}
