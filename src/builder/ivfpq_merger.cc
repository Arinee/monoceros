#include <fstream>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>
#include <chrono>
#include <thread>
#include "ivfpq_merger.h"
#include <framework/utility/file.h>
#include <framework/index_error.h>
#include <framework/index_meta.h>
#include <framework/vector_holder.h>
#include <framework/index_package.h>
#include <framework/index_logger.h>
#include <framework/utility/buffer_sampler.h>
#include <framework/utility/bitset_helper.h>
#include "index/query_distance_matrix.h"
#include "utils/string_util.h"
#include "common/common_define.h"
#include "utils/hash_table.h"
#include "index/pq_index_error.h"

using namespace std;
namespace mercury {

int IvfpqMerger::Init(const IndexParams &params) {
    _segmentDir = params.getString(PARAM_PQ_BUILDER_INTERMEDIATE_PATH);
    if (_segmentDir == "") {
        _segmentDir = ".";
    }
    _segmentDir += "/" + to_string(getpid());


    return 0;
}

//! Cleanup Builder
int IvfpqMerger::Cleanup(void) {
    if (!File::RemovePath(_segmentDir.c_str())) {
        return IndexError_RemoveSegment;
    }
    return 0;
}

int IvfpqMerger::FeedIndex(const std::vector<std::string> &paths,
              const IndexStorage::Pointer &stg)
{
    for (size_t i = 0 ; i < paths.size(); ++i) {
        auto path = paths[i];
        auto handler = stg->open(path, false);
        if (!handler) {
            LOG_WARN("try open [%s] as a index directory.", path.c_str());
            handler = stg->open(path + "/" + PQ_INDEX_FILENAME, false);
            if (!handler) {
                return -1;
            }
        }
        IndexIvfpq::Pointer index = make_shared<IndexIvfpq>();
        if (!index->Load(move(handler))) {
            LOG_ERROR("load ivfpq index error. path: %s", path.c_str());
            return -1;
        }
        _indexes.push_back(index);
    }
    if (_indexes.empty()) {
        LOG_ERROR("loaded index list is empty");
        return -1;
    }

    _meta = *_indexes[0]->getIndexMeta();
    _resource = _indexes[0]->getCentroidResource();

    _mergedDocNum = 0;
    for (size_t i = 0; i < _indexes.size(); ++i) {
        auto index = _indexes[i];
        _mergedDocNum += index->get_doc_num();
    }
    LOG_INFO("Read merged doc num : %lu, segment count: %lu", _mergedDocNum, _indexes.size());

    return 0;
}


int IvfpqMerger::MergeIndex()
{
    if (_indexes.empty()) {
        LOG_ERROR("segment index list is empty");
        return -1;
    }

    if (!File::MakePath(_segmentDir.c_str())) {
        LOG_ERROR("make segment directory[%s] failed!", _segmentDir.c_str());
        return -1;
    }

    auto maxDocNum = _mergedDocNum;

    size_t coarseIndexSlotNum = (size_t)_indexes[0]->get_coarse_index()->getHeader()->slotNum;
    size_t indexCapacity = CoarseIndex::calcSize(coarseIndexSlotNum, maxDocNum);
    auto pkCapacity = ArrayProfile::CalcSize(maxDocNum, _indexes[0]->getPrimaryKeyProfile()->getHeader()->infoSize);
    auto productCapacity = ArrayProfile::CalcSize(maxDocNum, _indexes[0]->getPqCodeProfile()->getHeader()->infoSize);
    auto featureCapacity = ArrayProfile::CalcSize(maxDocNum, _indexes[0]->getFeatureProfile()->getHeader()->infoSize);
    auto idMapCapacity = HashTable<uint64_t, docid_t>::needMemSize(maxDocNum);

    // mmap output file
    bool res = indexMergeFile.create(string(_segmentDir + "/" + COMPONENT_COARSE_INDEX).c_str(), indexCapacity);
    coarseIndex = make_shared<CoarseIndex>();
    res &= coarseIndex->create(indexMergeFile.region(), indexMergeFile.region_size(),
                               coarseIndexSlotNum, maxDocNum);

    res &= pkMergeFile.create(string(_segmentDir + "/" + COMPONENT_PK_PROFILE).c_str(), pkCapacity);
    pkProfile = make_shared<ArrayProfile>();
    res &= pkProfile->create(pkMergeFile.region(), pkMergeFile.region_size(),
                             _indexes[0]->getPrimaryKeyProfile()->getHeader()->infoSize);

    res &= productMergeFile.create(string(_segmentDir + "/" + COMPONENT_PRODUCT_PROFILE).c_str(), productCapacity);
    pqcodeProfile = make_shared<ArrayProfile>();
    res &= pqcodeProfile->create(productMergeFile.region(), productMergeFile.region_size(),
                                 _indexes[0]->getPqCodeProfile()->getHeader()->infoSize);

    res &= featureMergeFile.create(string(_segmentDir + "/" + COMPONENT_FEATURE_PROFILE).c_str(), featureCapacity);
    featureProfile = make_shared<ArrayProfile>();
    res &= featureProfile->create(featureMergeFile.region(), featureMergeFile.region_size(),
                                  _indexes[0]->getFeatureProfile()->getHeader()->infoSize);

    res &= idMapMergeFile.create(string(_segmentDir + "/" + COMPONENT_IDMAP).c_str(), idMapCapacity);
    idMapPtr = make_shared<HashTable<uint64_t, docid_t>>();
    int ret = idMapPtr->mount(reinterpret_cast<char *>(idMapMergeFile.region()), idMapMergeFile.region_size(),
                              maxDocNum, true);
    if (!res || ret < 0) {
        LOG_ERROR("mmap output file error!");
        return IndexError_CreateIndex;
    }

    docid_t globalId = 0;
    docid_t docId = INVALID_DOCID;
    int64_t slotNum = _resource->getLeafCentroidNum();
    for (int64_t i = 0; i < slotNum; ++i) {
        for (size_t j = 0; j < _indexes.size(); ++j) {
            auto iter = _indexes[j]->get_coarse_index()->search(i);
            while ((docId = iter.next()) != INVALID_DOCID) {
                uint64_t pk = *reinterpret_cast<const uint64_t *>(_indexes[j]->getPrimaryKeyProfile()->getInfo(docId));
                docid_t tempId = INVALID_DOCID;
                if (idMapPtr->find(pk, tempId)) {
                    LOG_WARN("insert duplicated doc with key[%lu]", pk);
                    continue;
                }
                res = coarseIndex->addDoc(i, globalId);
                if (!res) {
                    LOG_WARN("insert doc[%d] error with id[%d] from segment[%zd]", globalId, docId, j);
                    continue;
                }
                res = pkProfile->insert(globalId, &pk);
                res &= pqcodeProfile->insert(globalId, _indexes[j]->getPqCodeProfile()->getInfo(docId));
                res &= featureProfile->insert(globalId, _indexes[j]->getFeatureProfile()->getInfo(docId));
                ret = idMapPtr->insert(pk, globalId);
                if (!res || ret != 0) {
                    LOG_ERROR("insert profile info error with id[%d]", globalId);
                }
                globalId += 1;
            }
        }
    }


    return 0;
}

int IvfpqMerger::DumpIndex(const std::string &prefix, const IndexStorage::Pointer &stg)
{
    LOG_INFO("Dump file with index package.");

    IndexPackage package;
    string metaData;
    _meta.serialize(&metaData);
    package.emplace(COMPONENT_FEATURE_META, metaData.data(), metaData.size());

    string roughMatrix, integrateMatrix;
    _resource->dumpRoughMatrix(roughMatrix);
    _resource->dumpIntegrateMatrix(integrateMatrix);
    package.emplace(COMPONENT_ROUGH_MATRIX, roughMatrix.data(), roughMatrix.size());
    package.emplace(COMPONENT_INTEGRATE_MATRIX, integrateMatrix.data(), integrateMatrix.size());

    package.emplace(COMPONENT_COARSE_INDEX, indexMergeFile.region(), indexMergeFile.region_size());
    package.emplace(COMPONENT_PK_PROFILE, pkMergeFile.region(), pkMergeFile.region_size());
    package.emplace(COMPONENT_PRODUCT_PROFILE, productMergeFile.region(), productMergeFile.region_size());
    package.emplace(COMPONENT_FEATURE_PROFILE, featureMergeFile.region(), featureMergeFile.region_size());
    package.emplace(COMPONENT_IDMAP, idMapMergeFile.region(), idMapMergeFile.region_size());

    size_t delMapSize = BitsetHelper::CalcBufferSize(_mergedDocNum);
    char *delMapData = new char[delMapSize];
    memset(delMapData, 0, delMapSize);
    package.emplace(COMPONENT_DELETEMAP, delMapData, delMapSize);

    string fileName = PQ_INDEX_FILENAME;
    if (fileName.empty()) {
        LOG_ERROR("Dump file name is empty string!");
        return -1;
    }
    if (!package.dump(prefix + '/' + fileName, stg, false)) {
        LOG_ERROR("Dump file failed!");
        return -1;
    }
    return 0;
}

}; // namespace mercury
