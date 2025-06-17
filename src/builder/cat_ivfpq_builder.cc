#include <fstream>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>
#include <chrono>
#include <thread>
#include "cat_ivfpq_builder.h"
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



CatIvfpqBuilder::~CatIvfpqBuilder()  {
    // free memory
    _coarseBase.reset();
    _pkBase.reset();
    _catBase.reset();
    _productBase.reset();
    _featureBase.reset();
}

int CatIvfpqBuilder::Init(const IndexMeta &meta, const IndexParams &params) {
    _meta = meta;
    _params = params;

    _segment = 0;
    _segmentDir = _params.getString(PARAM_PQ_BUILDER_INTERMEDIATE_PATH);
    if (_segmentDir == "") {
        _segmentDir = ".";
    }
    _segmentDir += "/" + to_string(getpid());
    string rough_file = _params.getString(PARAM_ROUGH_MATRIX);
    string integrate_file = _params.getString(PARAM_INTEGRATE_MATRIX);

    //must has rough & integrate
    if (!rough_file.empty() && !integrate_file.empty()) {
        // check 
        if (!InitResource(rough_file, integrate_file)) {
            LOG_ERROR("init centroid resource failed");
            return -1;
        }
    } else { // get pq codebook failed
        LOG_DEBUG("can not find pqCodebook in index params, need train step!");
        return -1;
    }

    return 0;
}

//! Initialize Builder
bool CatIvfpqBuilder::InitResource(PQCodebook::Pointer pqCodebook) {
    // TODO check index meta of codebook
    if (!pqCodebook || !pqCodebook->checkValid(_meta)) {
        LOG_WARN("pqCodebook has wrong format");
        return false;
    }

    uint32_t roughElemSize = pqCodebook->getRoughCentroidSize();
    std::vector<uint32_t> layerPattern = pqCodebook->getLayerPattern();
    std::vector<uint32_t> layerInfo = pqCodebook->getLayerInfo();

    uint32_t integrateElemSize = pqCodebook->getIntegrateCentroidSize();
    uint32_t integrateFragmentNum = pqCodebook->getFragmentNum();
    uint32_t integrateCentroidNum = pqCodebook->getIntegrateCentroidNum();
    // process integrate codebook
    CentroidResource::RoughMeta roughMeta(roughElemSize, layerPattern.size(), layerPattern);
    CentroidResource::IntegrateMeta integrateMeta(integrateElemSize, integrateFragmentNum, integrateCentroidNum);
    _resource.reset(new CentroidResource);
    bool bret = _resource->create(roughMeta, integrateMeta);
    if (!bret) {
        LOG_WARN("Failed to create resource.");
        return false;
    }

    // process rough centroid
    for (uint32_t l = 0, sz = layerInfo.size(); l < sz; ++l) {
        uint32_t levelCentroidNum = layerInfo[l];
        for (uint32_t c = 0; c < levelCentroidNum; ++c) {
            const void *roughCentroid = pqCodebook->getRoughCentroid(l, c);
            if (!roughCentroid) {
                LOG_WARN("get rough centroid from codebook error.");
                return false;
            }
            if (!_resource->setValueInRoughMatrix(l, c, roughCentroid)) {
                LOG_WARN("set rough centroid into resource error.");
                return false;
            }
        }
    }

    // process integrate centroid
    for (uint32_t f = 0; f < integrateFragmentNum; ++f) {
        for (uint32_t c = 0; c < integrateCentroidNum; ++c) {
            const void *centroid = pqCodebook->getIntegrateCentroid(f, c);
            if (!centroid) {
                LOG_WARN("get integrate centroid from codebook error.");
                return false;
            }

            if (!_resource->setValueInIntegrateMatrix(f, c, centroid)) {
                LOG_WARN("resource set integrate centroid error");
                return false;
            }
        }
    }

    return true;
}

//! init from matrix
bool CatIvfpqBuilder::InitResource(const string &rough_file, const string &integrate_file) {
    _resource.reset(new CentroidResource);
    roughFile.open(rough_file.c_str(), true);
    integrateFile.open(integrate_file.c_str(), true);

    bool bret = _resource->init((void *) roughFile.region(),
                                roughFile.region_size(),
                                (void *) integrateFile.region(),
                                integrateFile.region_size());
    if (!bret) {
        LOG_ERROR("centroid resource init error");
        return false;
    }

    return true;
}

//! Cleanup Builder
int CatIvfpqBuilder::Cleanup() {
    // free memory
    _coarseBase.reset();
    _pkBase.reset();
    _productBase.reset();
    _featureBase.reset();

    if (!File::RemovePath(_segmentDir.c_str())) {
        return IndexError_RemoveSegment;
    }
    return 0;
}

int CatIvfpqBuilder::Train(const VectorHolder::Pointer & /* holder */)
{
    return 0;
}

bool CatIvfpqBuilder::IsFinish() {
    return _isRuned;
}

int CatIvfpqBuilder::DumpIndex(const std::string &prefix, const IndexStorage::Pointer &stg) {
    // write left to disk
    if (_globalId > 0) {
        LOG_DEBUG("left %lu data, will flush to disk", _globalId.load());
        if (false == flushWithAdjust()) {
            LOG_ERROR("Write segment[%zd] to disk failed", _segment);
            return IndexError_FlushSegment;
        }
    } else if (_globalId == 0) {
        LOG_ERROR("Get feature data failed!");
        return IndexError_NoFeatureFound;
    }

    if (_segmentList.size() == 0) {
        LOG_ERROR("segment list is empty, need not dump!");
        return IndexError_Runtime;
    }
    if (!stg.get()) {
        LOG_ERROR("get storage failed!");
        return IndexError_InvalidArgument;
    }
    if (!_resource) {
        LOG_ERROR("centroid resource is null, need train first or init resource with index params.");
        return IndexError_Runtime;
    }

    size_t segmentCount = _segmentList.size();
    vector<unique_ptr<MMapFile>> fileHolder;
    vector<CoarseIndex::Pointer> indexSegs;
    vector<ArrayProfile::Pointer> pkSegs;
    vector<ArrayProfile::Pointer> catSegs;
    vector<ArrayProfile::Pointer> productSegs;
    vector<ArrayProfile::Pointer> featureSegs;

    int64_t maxDocNum = loadSegments(fileHolder, indexSegs, pkSegs, catSegs, productSegs, featureSegs);
    if (maxDocNum < 0) {
        LOG_ERROR("load segments failed!");
        return IndexError_IndexLoaded;
    }
    LOG_DEBUG("Total doc count[%zd] with %zd segments.", maxDocNum, segmentCount);

    int64_t indexCapacity = CoarseIndex::calcSize(indexSegs[0]->getHeader()->slotNum, maxDocNum);
    int64_t pkCapacity = ArrayProfile::CalcSize(maxDocNum, pkSegs[0]->getHeader()->infoSize);
    int64_t catCapacity = ArrayProfile::CalcSize(maxDocNum, catSegs[0]->getHeader()->infoSize);
    int64_t productCapacity = ArrayProfile::CalcSize(maxDocNum, productSegs[0]->getHeader()->infoSize);
    int64_t featureCapacity = ArrayProfile::CalcSize(maxDocNum, featureSegs[0]->getHeader()->infoSize);
    uint64_t idMapCapacity = HashTable<uint64_t, docid_t>::needMemSize(maxDocNum);
    uint64_t keyCatMapCapacity = HashTable<key_t, cat_t, 1>::needMemSize(maxDocNum);

    // mmap output file
    mercury::MMapFile indexMergeFile, pkMergeFile, catMergeFile, keyCatMapMergeFile, catSetMergeFile, productMergeFile, featureMergeFile, idMapMergeFile;
    bool res = indexMergeFile.create(string(_segmentDir + "/" + COMPONENT_COARSE_INDEX).c_str(), indexCapacity);
    CoarseIndex::Pointer coarseIndex(new CoarseIndex);
    res &= coarseIndex->create(indexMergeFile.region(), indexMergeFile.region_size(),
                               indexSegs[0]->getHeader()->slotNum, maxDocNum);


    res &= pkMergeFile.create(string(_segmentDir + "/" + COMPONENT_PK_PROFILE).c_str(), pkCapacity);
    ArrayProfile::Pointer pkProfile(new ArrayProfile);
    res &= pkProfile->create(pkMergeFile.region(), pkMergeFile.region_size(),
                             pkSegs[0]->getHeader()->infoSize);

    res &= catMergeFile.create(string(_segmentDir + "/" + "cat.profile").c_str(), catCapacity);
    ArrayProfile::Pointer catProfile(new ArrayProfile);
    res &= catProfile->create(catMergeFile.region(), catMergeFile.region_size(),
            catSegs[0]->getHeader()->infoSize);

    res &= keyCatMapMergeFile.create(string(_segmentDir + "/" + CAT_COMPONENT_KEY_CAT_MAP).c_str(), keyCatMapCapacity);
    auto keyCatMapPtr = std::make_shared<HashTable<key_t, cat_t, 1>>();
    int ret = keyCatMapPtr->mount(reinterpret_cast<char *>(keyCatMapMergeFile.region()),keyCatMapMergeFile.region_size(), maxDocNum, true);

    res &= productMergeFile.create(string(_segmentDir + "/" + COMPONENT_PRODUCT_PROFILE).c_str(), productCapacity);
    ArrayProfile::Pointer pqcodeProfile(new ArrayProfile);
    res &= pqcodeProfile->create(productMergeFile.region(), productMergeFile.region_size(),
                                 productSegs[0]->getHeader()->infoSize);

    res &= featureMergeFile.create(string(_segmentDir + "/" + COMPONENT_FEATURE_PROFILE).c_str(), featureCapacity);
    ArrayProfile::Pointer featureProfile(new ArrayProfile);
    res &= featureProfile->create(featureMergeFile.region(), featureMergeFile.region_size(),
                                  featureSegs[0]->getHeader()->infoSize);

    res &= idMapMergeFile.create(string(_segmentDir + "/" + COMPONENT_IDMAP).c_str(), idMapCapacity);
    shared_ptr<HashTable<uint64_t, docid_t>> idMapPtr(new HashTable<uint64_t, docid_t>());
    ret = idMapPtr->mount(reinterpret_cast<char *>(idMapMergeFile.region()), idMapMergeFile.region_size(),
                              maxDocNum, true);

    if (!res || ret < 0) {
        LOG_ERROR("mmap output file error!");
        return IndexError_CreateIndex;
    }

    std::unordered_set<cat_t> catSet;
    docid_t globalId = 0;
    docid_t docId = INVALID_DOCID;
    int64_t slotNum = _resource->getLeafCentroidNum();
    for (int64_t i = 0; i < slotNum; ++i) {
        for (size_t j = 0; j < segmentCount; ++j) {
            auto iter = indexSegs[j]->search(i);
            while ((docId = iter.next()) != INVALID_DOCID) {
                uint64_t pk = *reinterpret_cast<const uint64_t *>(pkSegs[j]->getInfo(docId));
                uint64_t cat = *reinterpret_cast<const uint64_t*>(catSegs[j]->getInfo(docId));
                docid_t tempId = INVALID_DOCID;
                if (idMapPtr->find(pk, tempId)) {
                    LOG_WARN("insert duplicated doc with key[%lu]", pk);
                    continue;
                }
                cat_t tempCat = INVALID_CAT_ID;
                if (keyCatMapPtr->find(pk, tempCat)) {
                    LOG_WARN("key-cat insert duplicated doc with key[%lu]", pk);
                    continue;
                }
                res = coarseIndex->addDoc(i, globalId);
                if (!res) {
                    LOG_WARN("insert doc[%d] error with id[%d] from segment[%zd]", globalId, docId, j);
                    continue;
                }
                res = pkProfile->insert(globalId, &pk);
                res &= pqcodeProfile->insert(globalId, productSegs[j]->getInfo(docId));
                res &= featureProfile->insert(globalId, featureSegs[j]->getInfo(docId));
                ret = idMapPtr->insert(pk, globalId);
                ret = keyCatMapPtr->insert(pk, cat);
                if (!res || ret != 0) {
                    LOG_ERROR("insert profile info error with id[%d]", globalId);
                }
                catSet.insert(cat);
                globalId += 1;
            }
        }
    }

    uint64_t catSetCapacity = HashTable<cat_t, cat_t>::needMemSize(catSet.size());
    res &= catSetMergeFile.create(string(_segmentDir + "/" + CAT_COMPONENT_CAT_SET).c_str(), catSetCapacity);
    shared_ptr<HashTable<cat_t, cat_t>> catSetPtr(new HashTable<cat_t, cat_t>());
    ret = catSetPtr->mount(reinterpret_cast<char *>(catSetMergeFile.region()), catSetMergeFile.region_size(), catSet.size(), true);
    if (!res || ret < 0) {
        LOG_ERROR("mmap output file error!");
        return IndexError_CreateIndex;
    }
    for(const auto& e : catSet) {
        if (catSetPtr->insert(e, 0x1) != 0) {
            LOG_ERROR("failed to add cat into cat set");
        }
    }
    LOG_DEBUG("Dump file with index package.");
    if (!writeIndexPackage(maxDocNum, prefix, stg, indexMergeFile, pkMergeFile,
                            keyCatMapMergeFile, catSetMergeFile,
                           productMergeFile, featureMergeFile, idMapMergeFile)) {
        LOG_ERROR("Dump file with index package failed!");
        return IndexError_DumpPackageIndex;
    }
    return 0;
}

list<Closure::Pointer> CatIvfpqBuilder::JobSplit(const VectorHolder::Pointer &holder) {
    _isRuned = true;
    list<Closure::Pointer> job_list;
    if (!_resource) {
        LOG_ERROR("centroid resource is null, need train first or init resource with index params.");
        return job_list;
    }

    size_t elemSize = holder->sizeofElement();
    size_t memQuota = _params.getUint64(PARAM_GENERAL_BUILDER_MEMORY_QUOTA);
    if (memQuota == 0) {
        memQuota = 10L * 1024L * 1024L * 1024L;
        LOG_ERROR("memory quota is not set, using default 10GB memory quota.");
    }
    size_t elemCount = memQuota2DocCount(memQuota, elemSize);
    if (elemCount < MIN_BUILD_COUNT) {
        LOG_ERROR("memory quota is wrong or too small");
        return job_list;
    }
    LOG_DEBUG("can build doc count[%zd] every time with memory quota[%zd]", elemCount, memQuota);
    size_t dimension = holder->dimension();
    LOG_DEBUG("element size[%zd], element cout[%zd], dimension[%zd]", elemSize, elemCount, dimension);

    int res = initProfile(elemCount, elemSize);
    if (res != 0) {
        LOG_ERROR("Init profiles error!");
        return job_list;
    }
    LOG_DEBUG("Init profiles success!");

    _globalId = 0;
    auto iter = holder->createIterator();
    if (!iter) {
        LOG_ERROR("Create iterator for holder failed");
        return job_list;
    }

    for (; iter->isValid();) {
        //if (job_list.size() >= 1) {
        //    //this_thread::sleep_for(chrono::microseconds(1));
        //    break;
        //}
        shared_ptr<char> data(new char[elemSize], std::default_delete<char[]>());
        memcpy(data.get(), reinterpret_cast<const char *>(iter->data()), elemSize);
        uint64_t key = iter->key();
        //cout << "key:" << key << ",elemSize:" << elemSize << endl;
        job_list.push_back(Closure::New(this, &CatIvfpqBuilder::singleTaskWithoutLabels, iter->cat(), key, data));
        iter->next();
    }

    return job_list;
}

void CatIvfpqBuilder::singleTaskWithoutLabels(cat_t cat_, uint64_t key, shared_ptr<char> data) {
    // calculate the code feature
    QueryDistanceMatrix qdm(_meta, _resource.get());
    // use 10% as level scan limit
    vector<size_t> levelScanLimit;
    for (size_t i = 0; i < _resource->getRoughMeta().levelCnt - 1; ++i) {
        levelScanLimit.push_back(_resource->getRoughMeta().centroidNums[i] / 10);
    }
    bool bres = qdm.init(data.get(), levelScanLimit, true);
    if (!bres) {
        LOG_ERROR("Calcualte QDM failed!");
        return;
    }
    vector<uint16_t> productLabels;
    if (!qdm.getQueryCodeFeature(productLabels)) {
        LOG_ERROR("get query codefeature failed!");
        return;
    }

    // calculate the rough label
    auto &centroids = qdm.getCentroids();
    int32_t roughLabel = centroids.top().index;
    if (roughLabel < 0) {
        LOG_ERROR("Calculate rough label failed[%d] with doc[%lu]!", roughLabel, key);
        return;
    }
    if (!doSingleBuild(cat_, key, data, roughLabel, productLabels)) {
        LOG_ERROR("Failed to do single build for key[%lu]\n", key);
    }
    return;
}

bool CatIvfpqBuilder::doSingleBuild(cat_t cat_, uint64_t key,
                                 shared_ptr<char> data,
                                 int32_t roughLabel,
                                 const vector<uint16_t> &productLabels) {
    lock_guard<mutex> lock(_docidLock);
    if (_featureProfile.isFull()) {
        LOG_DEBUG("segment[%zd] is full[%ld], flush to disk and create new segment.",
                  _segment, _featureProfile.getHeader()->usedDocNum);
        // write to disk
        if (false == flushWithAdjust()) {
            LOG_ERROR("Write segment[%zd] to disk failed", _segment);
            return false;
        }
        _globalId = 0;
    }
    docid_t docid = _globalId;
    if (docid % MIN_BUILD_COUNT == 0) {
        LOG_DEBUG("build processed item: %u, usedDoc[%ld], maxDoc[%ld]", docid,
                  _featureProfile.getHeader()->usedDocNum, _featureProfile.getHeader()->maxDocNum);
    }

    bool bret = _coarseIndex.addDoc(roughLabel, docid);
    if (!bret) {
        LOG_ERROR("Add doc[%lu] to coarse index failed!", key);
        return false;
    }

    bret = _featureProfile.insert(docid, data.get());
    bret &= _pqcodeProfile.insert(docid, &productLabels[0]);
    bret &= _pkProfile.insert(docid, &key);
    bret &= _catProfile.insert(docid, &cat_);
    if (!bret) {
        LOG_ERROR("Add doc[%lu] to profiles failed with docid [%u]!", key, docid);
        return false;
    }
    _globalId += 1;

    return true;
}

bool CatIvfpqBuilder::flushSegment() {
    // make dump path
    string segmentPath = _segmentDir + "/segment_" + to_string(_segment);
    if (!File::MakePath(segmentPath.c_str())) {
        LOG_ERROR("make segment directory[%s] failed!", segmentPath.c_str());
        return false;
    }

    // write index to disk as segment
    bool res = _coarseIndex.dump(segmentPath + "/" + COMPONENT_COARSE_INDEX);
    res &= _pkProfile.dump(segmentPath + "/" + COMPONENT_PK_PROFILE);
    res &= _pqcodeProfile.dump(segmentPath + "/" + COMPONENT_PRODUCT_PROFILE);
    res &= _featureProfile.dump(segmentPath + "/" + COMPONENT_FEATURE_PROFILE);
    if (!res) {
        return false;
    }

    _coarseIndex.reset();
    _pkProfile.reset();
    _pqcodeProfile.reset();
    _featureProfile.reset();

    _segment += 1;
    _segmentList.push_back(segmentPath);

    LOG_DEBUG("Write segment to disk done: %s", segmentPath.c_str());

    return true;
}

bool CatIvfpqBuilder::flushWithAdjust() {
    // make dump path
    string segmentPath = _segmentDir + "/segment_" + to_string(_segment);
    if (!File::MakePath(segmentPath.c_str())) {
        LOG_ERROR("make segment directory[%s] failed!", segmentPath.c_str());
        return false;
    }

    int64_t slotNum = _resource->getLeafCentroidNum();
    size_t pkInfoSize = _pkProfile.getHeader()->infoSize;
    size_t catInfoSize = _catProfile.getHeader()->infoSize;
    size_t pqcodeSize = _pqcodeProfile.getHeader()->infoSize;
    size_t featureInfoSize = _featureProfile.getHeader()->infoSize;
    int64_t totalDocNum = _pkProfile.getHeader()->usedDocNum;

    ofstream indexStream(segmentPath + "/" + COMPONENT_COARSE_INDEX, ofstream::binary);
    ofstream pkStream(segmentPath + "/" + COMPONENT_PK_PROFILE, ofstream::binary);
    ofstream catStream(segmentPath + "/" + "cat.profile", ofstream::binary);
    ofstream productStream(segmentPath + "/" + COMPONENT_PRODUCT_PROFILE, ofstream::binary);
    ofstream featureStream(segmentPath + "/" + COMPONENT_FEATURE_PROFILE, ofstream::binary);

    // write headers
    size_t headerSize = sizeof(ArrayProfile::Header);
    char headerData[headerSize];
    ArrayProfile::Header *header = reinterpret_cast<ArrayProfile::Header *>(headerData);
    memcpy(headerData, _pkProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, pkInfoSize);
    pkStream.write(headerData, headerSize);

    memcpy(headerData, _catProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, catInfoSize);
    catStream.write(headerData, headerSize);

    memcpy(headerData, _pqcodeProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, pqcodeSize);
    productStream.write(headerData, headerSize);
    memcpy(headerData, _featureProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, featureInfoSize);
    featureStream.write(headerData, headerSize);

    size_t indexSize = CoarseIndex::calcSize(slotNum, totalDocNum);
    shared_ptr<char> data(new char[indexSize], std::default_delete<char[]>());
    if (!data) {
        LOG_ERROR("alloc index memory error!");
        return false;
    }
    CoarseIndex index;
    bool res = index.create(data.get(), indexSize, slotNum, totalDocNum);
    if (!res) {
        LOG_ERROR("create index error!");
        return false;
    }
    docid_t docId = INVALID_DOCID;
    docid_t newId = 0;
    for (int64_t i = 0; i < slotNum; ++i) {
        auto iter = _coarseIndex.search(i);
        while ((docId = iter.next()) != INVALID_DOCID) {
            pkStream.write(reinterpret_cast<const char *>(_pkProfile.getInfo(docId)), pkInfoSize);
            catStream.write(reinterpret_cast<const char*>(_catProfile.getInfo(docId)), catInfoSize);
            productStream.write(reinterpret_cast<const char *>(_pqcodeProfile.getInfo(docId)), pqcodeSize);
            featureStream.write(reinterpret_cast<const char *>(_featureProfile.getInfo(docId)), featureInfoSize);
            res = index.addDoc(i, newId++);
            if (!res) {
                LOG_ERROR("index rewrite docid error!");
                return false;
            }
        }
    }
    indexStream.write(data.get(), indexSize);

    _coarseIndex.reset();
    _pkProfile.reset();
    _catProfile.reset();
    _pqcodeProfile.reset();
    _featureProfile.reset();

    _segment += 1;
    _segmentList.push_back(segmentPath);
    LOG_DEBUG("Write segment to disk done: %s", segmentPath.c_str());

    return true;
}

int CatIvfpqBuilder::initProfile(size_t elemCount, size_t elemSize) {
    // Create CoarseIndex
    int64_t slotNum = _resource->getLeafCentroidNum();
    int64_t capacity = CoarseIndex::calcSize(slotNum, elemCount);
    _coarseBase.reset(new char[capacity], std::default_delete<char[]>());
    memset(_coarseBase.get(), 0, capacity);
    bool bret = _coarseIndex.create(_coarseBase.get(), capacity, slotNum, elemCount);
    if (bret != true) {
        LOG_ERROR("create Coarse Index failed!");
        return -1;
    }

    // Create Profiles
    size_t pkCapacity = ArrayProfile::CalcSize(elemCount, sizeof(uint64_t));
    size_t catCapacity = ArrayProfile::CalcSize(elemCount, sizeof(uint64_t));
    _pkBase.reset(new char[pkCapacity], std::default_delete<char[]>());
    memset(_pkBase.get(), 0, pkCapacity);
    int res = _pkProfile.create(_pkBase.get(), pkCapacity, sizeof(uint64_t));
    if (res < 0) {
        LOG_ERROR("create PK Profile failed!");
        return -1;
    }

    _catBase.reset(new char[catCapacity], std::default_delete<char[]>());
    memset(_catBase.get(), 0, catCapacity);
    res = _catProfile.create(_catBase.get(), catCapacity, sizeof(uint64_t));
    if (res < 0) {
        LOG_ERROR("create CAT Profile failed!");
        return -1;
    }

    uint32_t fragmentNum = _resource->getIntegrateMeta().fragmentNum;
    size_t productSize = sizeof(uint16_t) * fragmentNum;
    size_t productCapacity = ArrayProfile::CalcSize(elemCount, productSize);
    _productBase.reset(new char[productCapacity], std::default_delete<char[]>());
    memset(_productBase.get(), 0, productCapacity);
    res = _pqcodeProfile.create(_productBase.get(), productCapacity, productSize);
    if (res < 0) {
        LOG_ERROR("create Product Profile failed!");
        return -1;
    }

    size_t featureCapacity = ArrayProfile::CalcSize(elemCount, elemSize);
    _featureBase.reset(new char[featureCapacity], std::default_delete<char[]>());
    memset(_featureBase.get(), 0, featureCapacity);
    res = _featureProfile.create(_featureBase.get(), featureCapacity, elemSize);
    if (res < 0) {
        LOG_ERROR("create Feature Profile failed!");
        return -1;
    }

    LOG_DEBUG("index capacity[%zd], pk capacity[%zd], product capacity[%zd], feature capacity[%zd]",
              capacity, pkCapacity, productCapacity, featureCapacity);
    return 0;
}

size_t CatIvfpqBuilder::memQuota2DocCount(size_t memQuota, size_t elemSize) {
    size_t memPreserved = 500L * 1024L * 1024L; // 500MB preserved memory size
    if (memQuota < memPreserved) {
        return 0;
    }

    int64_t slotNum = _resource->getLeafCentroidNum();
    uint32_t fragmentNum = _resource->getIntegrateMeta().fragmentNum;
    size_t productSize = sizeof(uint16_t) * fragmentNum;
    size_t elemCount = 0;
    size_t realMemUsed = 0;
    do {
        elemCount += MIN_BUILD_COUNT;
        realMemUsed = memPreserved;
        realMemUsed += CoarseIndex::calcSize(slotNum, elemCount);
        realMemUsed += ArrayProfile::CalcSize(elemCount, sizeof(uint64_t));
        realMemUsed += ArrayProfile::CalcSize(elemCount, productSize);
        realMemUsed += ArrayProfile::CalcSize(elemCount, elemSize);
    } while (realMemUsed <= memQuota);
    elemCount -= MIN_BUILD_COUNT;

    return elemCount;
}


int64_t CatIvfpqBuilder::loadSegments(vector<unique_ptr<mercury::MMapFile>> &fileHolder,
                                   vector<CoarseIndex::Pointer> &indexSegs,
                                   vector<ArrayProfile::Pointer> &pkSegs,
                                   vector<ArrayProfile::Pointer> &catSegs,
                                   vector<ArrayProfile::Pointer> &productSegs,
                                   vector<ArrayProfile::Pointer> &featureSegs) {
    int64_t maxDocNum = 0;
    for (string segmentPath : _segmentList) {
        unique_ptr<MMapFile> indexFile(new MMapFile);
        unique_ptr<MMapFile> pkFile(new MMapFile);
        unique_ptr<MMapFile> catFile(new MMapFile);
        unique_ptr<MMapFile> productFile(new MMapFile);
        unique_ptr<MMapFile> featureFile(new MMapFile);

        indexFile->open(string(segmentPath + "/" + COMPONENT_COARSE_INDEX).c_str(), true);
        pkFile->open(string(segmentPath + "/" + COMPONENT_PK_PROFILE).c_str(), true);
        catFile->open(string(segmentPath + "/" + "cat.profile").c_str(), true);
        productFile->open(string(segmentPath + "/" + COMPONENT_PRODUCT_PROFILE).c_str(), true);
        featureFile->open(string(segmentPath + "/" + COMPONENT_FEATURE_PROFILE).c_str(), true);
        if (!indexFile->isValid() || !pkFile->isValid() || !productFile->isValid()
            || !featureFile->isValid() || !catFile->isValid() ) {
            LOG_ERROR("open segment[%s] failed!", segmentPath.c_str());
            return -1;
        }

        // make segment index
        CoarseIndex::Pointer segmentIndex(new CoarseIndex());
        bool res = segmentIndex->load(indexFile->region(), indexFile->region_size());
        indexSegs.emplace_back(segmentIndex);
        fileHolder.push_back(move(indexFile));

        // make segment pk
        ArrayProfile::Pointer segmentPK(new ArrayProfile());
        res = segmentPK->load(pkFile->region(), pkFile->region_size());
        pkSegs.emplace_back(segmentPK);
        fileHolder.push_back(move(pkFile));

        // make segment cat
        ArrayProfile::Pointer segmentCAT(new ArrayProfile());
        res = segmentCAT->load(catFile->region(), catFile->region_size());
        catSegs.emplace_back(segmentCAT);
        fileHolder.push_back(move(catFile));

        // make segment product
        ArrayProfile::Pointer segmentProduct(new ArrayProfile());
        res = segmentProduct->load(productFile->region(), productFile->region_size());
        productSegs.emplace_back(segmentProduct);
        fileHolder.push_back(move(productFile));

        // make segment feature
        ArrayProfile::Pointer segmentFeature(new ArrayProfile());
        res = segmentFeature->load(featureFile->region(), featureFile->region_size());
        featureSegs.emplace_back(segmentFeature);
        fileHolder.push_back(move(featureFile));

        if (!res) {
            LOG_ERROR("load segment[%s] as index failed", segmentPath.c_str());
            return -1;
        }

        maxDocNum += segmentPK->getHeader()->usedDocNum;
    }
    return maxDocNum;
}

bool CatIvfpqBuilder::writeIndexPackage(size_t maxDocNum,
                                     const string &prefix,
                                     const IndexStorage::Pointer &stg,
                                     const mercury::MMapFile &indexMergeFile,
                                     const mercury::MMapFile &pkMergeFile,
                                     const mercury::MMapFile &keyCatMapMergeFile,
                                     const mercury::MMapFile &catSetMergeFile,
                                     const mercury::MMapFile &productMergeFile,
                                     const mercury::MMapFile &featureMergeFile,
                                     const mercury::MMapFile &idMapMergeFile) {
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
    package.emplace(CAT_COMPONENT_KEY_CAT_MAP, keyCatMapMergeFile.region(), keyCatMapMergeFile.region_size());
    package.emplace(CAT_COMPONENT_CAT_SET, catSetMergeFile.region(), catSetMergeFile.region_size());

    size_t delMapSize = BitsetHelper::CalcBufferSize(maxDocNum);
    char *delMapData = new char[delMapSize];
    memset(delMapData, 0, delMapSize);
    package.emplace(COMPONENT_DELETEMAP, delMapData, delMapSize);

    string fileName = CAT_IVFPQ_INDEX_FILENAME;
    if (fileName.empty()) {
        LOG_ERROR("Dump file name is empty string!");
        return false;
    }
    if (!package.dump(prefix + '/' + fileName, stg, false)) {
        LOG_ERROR("Dump file failed!");
        return false;
    }
    return true;
}

int CatIvfpqBuilder::BuildIndex(const VectorHolder::Pointer &holder) {
    if (!holder.get()) {
        return IndexError_InvalidArgument;
    }

    if (!_meta.isMatched(*holder)) {
        return mercury::IndexError_UnmatchedMeta;
    }
    // use graph_run_engine
    size_t threadCount = _params.getUint64(PARAM_GENERAL_BUILDER_THREAD_COUNT);

    MuitThreadBatchWorkflow thread_work_engine(threadCount);
    thread_work_engine.Init(std::bind(&CatIvfpqBuilder::IsFinish, this),
                            std::bind(&CatIvfpqBuilder::JobSplit, this, holder));
    thread_work_engine.Run();

    return 0;
}

}; //namespace mercury
