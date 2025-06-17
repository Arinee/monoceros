#ifndef __COMMON_PARAMS_DEFINE_H__
#define __COMMON_PARAMS_DEFINE_H__

#include <string>

namespace mercury {

// General
const std::string PARAM_GENERAL_BUILDER_MEMORY_QUOTA("mercury.general.builder.memory_quota");
const std::string PARAM_GENERAL_BUILDER_THREAD_COUNT("mercury.general.builder.thread_count");
const std::string PARAM_GENERAL_BUILDER_TRAIN_SAMPLE_COUNT("mercury.general.builder.train_sample_count");
const std::string PARAM_GENERAL_SEARCHER_SEARCH_METHOD("mercury.general.searcher.search_method");
const std::string PARAM_GENERAL_SEARCHER_BUILD_IDMAP("mercury.general.searcher.build_idmap");
const std::string PARAM_GENERAL_SEARCHER_INCR_SEGMENT_PATH("mercury.general.searcher.incr_segment_path");
const std::string PARAM_GENERAL_SEARCHER_INCR_SEGMENT_DOC_NUM("mercury.general.searcher.incr_segment_doc_number");

// Trainer
const std::string PARAM_ROUGH_MATRIX("mercury.general.trainer.rough_matrix");
const std::string PARAM_INTEGRATE_MATRIX("mercury.general.trainer.integrate_matrix");

// Linear
const std::string PARAM_LINEAR_BUILDER_FILENAME("mercury.linear.builder.filename");
const std::string PARAM_LINEAR_SEARCHER_FASTLOAD("mercury.linear.searcher.fastload");
const std::string PARAM_LINEAR_SEARCHER_THREADS("mercury.linear.searcher.threads");
const std::string PARAM_LINEAR_SEARCHER_FILENAME("mercury.linear.searcher.filename");
const std::string PARAM_LINEAR_SEARCHER_GPU_DEVICE_NO("mercury.linear.searcher.gpu_device_no");

// HC
const std::string PARAM_HC_COMMON_BASIC_BLOCK_SIZE("mercury.hc.common.basic_block_size");
const std::string PARAM_HC_COMMON_COMBO_FILE("mercury.hc.common.combo_file");
const std::string PARAM_HC_COMMON_LEVEL_CNT("mercury.hc.common.level_cnt");
const std::string PARAM_HC_COMMON_MAX_DOC_CNT("mercury.hc.common.max_doc_cnt");
const std::string PARAM_HC_COMMON_LEAF_CENTROID_NUM("mercury.hc.common.leaf_centroid_num");

const std::string PARAM_HC_BUILDER_CENTROID_INDEX("mercury.hc.builder.centroid_index");
const std::string PARAM_HC_BUILDER_BUILD_IN_MEMORY("mercury.hc.builder.build_in_memory");
const std::string PARAM_HC_BUILDER_MEMORY_QUOTA("mercury.hc.builder.memory_quota");
const std::string PARAM_HC_BUILDER_CENTROID_NUM_IN_LEVEL_PREFIX("mercury.hc.builder.num_in_level_");
const std::string PARAM_HC_BUILDER_BASIC_PATH("mercury.hc.builder.basic_path");
const std::string PARAM_HC_BUILDER_WORK_QUEUE_SIZE("mercury.hc.builder.work_queue_size");
const std::string PARAM_HC_BUILDER_CENTROIDS_NUM("mercury.hc.builder.centroids_num");

const std::string PARAM_HC_SEARCHER_SCAN_NUM_IN_LEVEL_PREFIX("mercury.hc.searcher.scan_num_in_level_");
const std::string PARAM_HC_SEARCHER_MAX_SCAN_NUM("mercury.hc.searcher.max_scan_num");
const std::string PARAM_HC_SEARCHER_TOPK("mercury.hc.searcher.topk");
const std::string PARAM_HC_SEARCHER_MAX_DISTANCE("mercury.hc.searcher.max_distance");

// PQ
const std::string PARAM_PQ_BUILDER_TRAIN_COARSE_CENTROID_NUM("mercury.pq.builder.train_coarse_centroid_number");
const std::string PARAM_PQ_BUILDER_TRAIN_PRODUCT_CENTROID_NUM("mercury.pq.builder.train_product_centroid_number");
const std::string PARAM_PQ_BUILDER_TRAIN_FRAGMENT_NUM("mercury.pq.builder.train_fragment_number");
const std::string PARAM_PQ_BUILDER_INTERMEDIATE_PATH("mercury.pq.builder.intermediate_path");
const std::string PARAM_PQ_BUILDER_CODEBOOK("mercury.pq.builder.codebook");
const std::string PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO("mercury.pq.searcher.coarse_scan_ratio");
const std::string PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM("mercury.pq.searcher.product_scan_number");

//mult-cat index
const std::string PARAM_MULT_CAT_INDEX_BUILDER_INTERMEDIATE_PATH("mercury.mult_cat_index.builder.intermediate_path");
const std::string PARAM_MULT_CAT_INDEX_BUILDER_CATE_NUM("mercury.mult_cat_index.builder.cate_num");

//FAISS
const std::string PARAM_FAISS_INDEX_NAME("mercury.faiss.index_name");
};

#endif // __COMMON_PARAMS_DEFINE_H__
