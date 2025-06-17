#ifndef COMMON_DEFINE_H_
#define COMMON_DEFINE_H_

#include <string>
#include <inttypes.h>
#include <limits>
#include <climits>
#include "params_define.h"

namespace mercury
{
//file path separator
const std::string SEPARATOR("/");
//graph using idx_t, as graph size can't exceed uint32_t
using docid_t = uint32_t;
using segid_t = uint32_t;
//! high 32 is segmentid, low 32 is docid
using gloid_t = uint64_t;
using idx_t = uint32_t;
using key_t = uint64_t;
using cat_t = std::uint64_t;
using slot_t = std::int32_t;

#define GET_GLOID(segid, docid) (((gloid_t)segid << 32L) | docid)
#define GET_SEGID(gloid) ((segid_t)(gloid >> 32L)) 
#define GET_DOCID(gloid) ((docid_t)(gloid & 0xFFFFFFFFul))

const uint64_t INVALID_OFFSET = std::numeric_limits<uint64_t>::max();
const key_t INVALID_KEY = std::numeric_limits<key_t>::max();
const docid_t INVALID_DOCID = std::numeric_limits<docid_t>::max();
const docid_t INVALID_SEGID = std::numeric_limits<segid_t>::max();
const gloid_t INVALID_GLOID = std::numeric_limits<gloid_t>::max();
const int SIZE_WRITE_ONE_TIME = 10 * 1024 * 1024;
const double BLOCK_USE_RATE = 0.5;
const auto INVALID_CAT_ID = std::numeric_limits<cat_t>::max();

//cat related names
const std::string CAT_FLAT_INDEX_FILENAME("cat_flat.indexes");
const std::string CAT_IVFFLAT_INDEX_FILENAME("cat_ivfflat.indexes");
const std::string CAT_IVFPQ_INDEX_FILENAME("cat_ivfpq.indexes");
const std::string CAT_COMPONENT_CAT_SLOT_MAP("cat_slot_map.dat");
const std::string CAT_COMPONENT_SLOT_DOC_INDEX("slot_doc_index.dat");
const std::string CAT_COMPONENT_KEY_CAT_MAP("key_cat_map.dat");
const std::string CAT_COMPONENT_CAT_SET("cat_set.dat");

//pq segment names
const std::string PQ_INDEX_FILENAME("ivfpq.indexes");
const std::string COMPONENT_FEATURE_META("feature_meta");
const std::string COMPONENT_COARSE_INDEX("coarse_index.dat");
const std::string COMPONENT_PK_PROFILE("pk_profile.dat");
const std::string COMPONENT_PRODUCT_PROFILE("product_profile.dat");
const std::string COMPONENT_FEATURE_PROFILE("feature_profile.dat");
const std::string COMPONENT_IDMAP("idmap.dat");
const std::string COMPONENT_DELETEMAP("delete_map.dat");
const std::string COMPONENT_ROUGH_MATRIX("rough_matrix");
const std::string COMPONENT_INTEGRATE_MATRIX("integrate_matrix");

//hc file names
const std::string COMBO_CLUSTERING_INDEX("combo_clustering_index");
const std::string INDEX_META("index_meta");
const std::string ID_MAP("id_map");
const std::string DELETE_MAP("delete_map");

//graph file names
const std::string COMBO_GRAPH_INDEX("combo_graph_index");

//graph type
const std::string HNSW_GRAPH("hnsw");
const std::string KGRAPH_GRAPH("kgraph");
const std::string NSG_GRAPH("nsg");
const std::string CLUSTERING_GRAPH("clustering");

//flat name
const std::string FLAT_INDEX_FILENAME("flat.indexes");
//PQFalt name
const std::string PQFLAT_INDEX_FILENAME("pqflat.indexes");
//ivfflat name
const std::string IVFFLAT_INDEX_FILENAME("ivfflat.indexes");
//cate inverted file name
const std::string INVERTED_INDEX_FILENAME("catinverted.indexes");
#define likely(x) __builtin_expect(!!(x), 1) 
#define unlikely(x) __builtin_expect(!!(x), 0)

#define DELETE_AND_SET_NULL(x) \
    do {                                        \
        if ((x) != NULL) {                      \
            delete (x);                         \
            (x) = NULL;                         \
        }                                       \
    } while (false)

}

#define GENERATE_RETURN_EMPTY_INDEX(class_name) \
virtual Index* CloneEmptyIndex()\
{\
    return new class_name(); \
}

// error code define index corresponding
const int SEGMENT_FULL_ERROR = -1;
const int INDEX_NOT_FOUND = -2;
const int INDEX_ALREADY_EXIST = -3;
const int PROFILE_NOT_FOUND = -4;
const int PROFILE_DEL_ERROR = -5;
const int PROFILE_ADD_ERROR = -6;


// index params define
const std::string kBuildDocNumKey("build_doc_num_key");
const std::string kFeatureInfoSizeKey("feature_info_size_key");
const std::string kProductCodeSizeKey("product_code_size_key");
const std::string kDumpDirPathKey("dump_dir_path_key");

// mult cate index
const std::string COMPONENT_CATEIDMAP("cateidmap.dat");

#endif //COMMON_DEFINE_H_
