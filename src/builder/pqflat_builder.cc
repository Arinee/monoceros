#include <fstream>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>
#include <chrono>
#include <thread>
#include "pqflat_builder.h"
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

PqflatBuilder::~PqflatBuilder()  {
    // free memory
    _pkBase.reset();
    _productBase.reset();
    _featureBase.reset();
}


int PqflatBuilder::Init(const IndexMeta &meta, const IndexParams &params) {
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
        LOG_DEBUG("can not find centroid file in index params, need train step!");
        return -1;
    }
    return 0;
}

//! init from matrix
bool PqflatBuilder::InitResource(const string &rough_file, const string &integrate_file) {
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

int PqflatBuilder::Train(const VectorHolder::Pointer & /* holder */)
{
    return 0;
}

//! Cleanup Builder
int PqflatBuilder::Cleanup() {
    if (!File::RemovePath(_segmentDir.c_str())) {
        return IndexError_RemoveSegment;
    }
    return 0;
}

int PqflatBuilder::DumpIndex(const std::string &prefix, const IndexStorage::Pointer &stg) {
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
    vector<ArrayProfile::Pointer> pkSegs;
    vector<ArrayProfile::Pointer> productSegs;
    vector<ArrayProfile::Pointer> featureSegs;

    int64_t maxDocNum = loadSegments(fileHolder, pkSegs, productSegs, featureSegs);
    if (maxDocNum < 0) {
        LOG_ERROR("load segments failed!");
        return IndexError_IndexLoaded;
    }
    LOG_DEBUG("Total doc count[%zd] with %zd segments.", maxDocNum, segmentCount);

    int64_t pkCapacity = ArrayProfile::CalcSize(maxDocNum, pkSegs[0]->getHeader()->infoSize);
    int64_t productCapacity = ArrayProfile::CalcSize(maxDocNum, productSegs[0]->getHeader()->infoSize);
    int64_t featureCapacity = ArrayProfile::CalcSize(maxDocNum, featureSegs[0]->getHeader()->infoSize);
    uint64_t idMapCapacity = HashTable<uint64_t, docid_t>::needMemSize(maxDocNum);

    // mmap output file
    mercury::MMapFile pkMergeFile, productMergeFile, featureMergeFile, idMapMergeFile;

    bool res = pkMergeFile.create(string(_segmentDir + "/" + COMPONENT_PK_PROFILE).c_str(), pkCapacity);
    ArrayProfile::Pointer pkProfile(new ArrayProfile);
    res &= pkProfile->create(pkMergeFile.region(), pkMergeFile.region_size(),
                             pkSegs[0]->getHeader()->infoSize);

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
    int ret = idMapPtr->mount(reinterpret_cast<char *>(idMapMergeFile.region()), idMapMergeFile.region_size(),
                              maxDocNum, true);

    if (!res || ret < 0) {
        LOG_ERROR("mmap output file error!");
        return IndexError_CreateIndex;
    }

    docid_t globalId = 0;
    docid_t docId = INVALID_DOCID;
    for (size_t i = 0; i < segmentCount; ++i) {
        for (int j = 0; j < pkSegs[i]->getHeader()->usedDocNum; ++j) {
            docId = j;
            uint64_t pk = *reinterpret_cast<const uint64_t *>(pkSegs[i]->getInfo(docId));
            docid_t tempId = INVALID_DOCID;
            if (idMapPtr->find(pk, tempId)) {
                LOG_WARN("insert duplicated doc with key[%lu]", pk);
                continue;
            }
            res = pkProfile->insert(globalId, &pk);
            res &= pqcodeProfile->insert(globalId, productSegs[i]->getInfo(docId));
            res &= featureProfile->insert(globalId, featureSegs[i]->getInfo(docId));
            ret = idMapPtr->insert(pk, globalId);
            if (!res || ret != 0) {
                LOG_ERROR("insert profile info error with id[%d]", globalId);
            }
            globalId += 1;

        }
    }

    LOG_DEBUG("Dump file with index package.");
    if (!writeIndexPackage(maxDocNum, prefix, stg, pkMergeFile,
                           productMergeFile, featureMergeFile, idMapMergeFile)) {
        LOG_ERROR("Dump file with index package failed!");
        return IndexError_DumpPackageIndex;
    }
    return 0;
}

list<Closure::Pointer> PqflatBuilder::JobSplit(const VectorHolder::Pointer &holder) {
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
        shared_ptr<char> data(new char[elemSize], std::default_delete<char[]>());
        memcpy(data.get(), reinterpret_cast<const char *>(iter->data()), elemSize);
        uint64_t key = iter->key();
        job_list.push_back(Closure::New(this, &PqflatBuilder::singleTaskCalcCode, key, data));
        iter->next();
    }

    return job_list;
}

void PqflatBuilder::singleTaskCalcCode(uint64_t key, shared_ptr<char> data) {
    // calculate the code feature
    QueryDistanceMatrix qdm(_meta, _resource.get());
    qdm.setWithCodeFeature(true);
    bool bres = qdm.computeDistanceMatrix(data.get());
    if (!bres) {
        LOG_ERROR("qdm ComputeDistanceMatrix failed!");
        return;
    }

    vector<uint16_t> productLabels;
    if (!qdm.getQueryCodeFeature(productLabels)) {
        LOG_ERROR("get query codefeature failed!");
        return;
    }

    if (!doSingleBuild(key, data, productLabels)) {
        LOG_ERROR("Failed to do single build for key[%lu]\n", key);
    }
    return;
}

bool PqflatBuilder::doSingleBuild(uint64_t key,
                                  shared_ptr<char> data,
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

    bool bret = _featureProfile.insert(docid, data.get());
    bret &= _pqcodeProfile.insert(docid, &productLabels[0]);
    bret &= _pkProfile.insert(docid, &key);
    if (!bret) {
        LOG_ERROR("Add doc[%lu] to profiles failed with docid [%u]!", key, docid);
        return false;
    }
    _globalId += 1;

    return true;
}

bool PqflatBuilder::flushWithAdjust() {
    // make dump path
    string segmentPath = _segmentDir + "/segment_" + to_string(_segment);
    if (!File::MakePath(segmentPath.c_str())) {
        LOG_ERROR("make segment directory[%s] failed!", segmentPath.c_str());
        return false;
    }

    size_t pkInfoSize = _pkProfile.getHeader()->infoSize;
    size_t pqcodeSize = _pqcodeProfile.getHeader()->infoSize;
    size_t featureInfoSize = _featureProfile.getHeader()->infoSize;
    int64_t totalDocNum = _pkProfile.getHeader()->usedDocNum;

    ofstream indexStream(segmentPath + "/" + COMPONENT_COARSE_INDEX, ofstream::binary);
    ofstream pkStream(segmentPath + "/" + COMPONENT_PK_PROFILE, ofstream::binary);
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
    memcpy(headerData, _pqcodeProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, pqcodeSize);
    productStream.write(headerData, headerSize);
    memcpy(headerData, _featureProfile.getHeader(), headerSize);
    header->maxDocNum = totalDocNum;
    header->capacity = ArrayProfile::CalcSize(totalDocNum, featureInfoSize);
    featureStream.write(headerData, headerSize);

    docid_t docId = INVALID_DOCID;
    for (int64_t i = 0; i < totalDocNum; ++i) {
        docId = i;
        pkStream.write(reinterpret_cast<const char *>(_pkProfile.getInfo(docId)), pkInfoSize);
        productStream.write(reinterpret_cast<const char *>(_pqcodeProfile.getInfo(docId)), pqcodeSize);
        featureStream.write(reinterpret_cast<const char *>(_featureProfile.getInfo(docId)), featureInfoSize);
    }

    _pkProfile.reset();
    _pqcodeProfile.reset();
    _featureProfile.reset();

    _segment += 1;
    _segmentList.push_back(segmentPath);
    LOG_DEBUG("Write segment to disk done: %s", segmentPath.c_str());

    return true;
}

int PqflatBuilder::initProfile(size_t elemCount, size_t elemSize) {
    // Create Profiles
    size_t pkCapacity = ArrayProfile::CalcSize(elemCount, sizeof(uint64_t));
    _pkBase.reset(new char[pkCapacity], std::default_delete<char[]>());
    memset(_pkBase.get(), 0, pkCapacity);
    int res = _pkProfile.create(_pkBase.get(), pkCapacity, sizeof(uint64_t));
    if (res < 0) {
        LOG_ERROR("create PK Profile failed!");
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

    LOG_DEBUG("index pk capacity[%zd], product capacity[%zd], feature capacity[%zd]",
              pkCapacity, productCapacity, featureCapacity);
    return 0;
}

size_t PqflatBuilder::memQuota2DocCount(size_t memQuota, size_t elemSize) {
    size_t memPreserved = 500L * 1024L * 1024L; // 500MB preserved memory size
    if (memQuota < memPreserved) {
        return 0;
    }

    uint32_t fragmentNum = _resource->getIntegrateMeta().fragmentNum;
    size_t productSize = sizeof(uint16_t) * fragmentNum;
    size_t elemCount = 0;
    size_t realMemUsed = 0;
    do {
        elemCount += MIN_BUILD_COUNT;
        realMemUsed = memPreserved;
        realMemUsed += ArrayProfile::CalcSize(elemCount, sizeof(uint64_t));
        realMemUsed += ArrayProfile::CalcSize(elemCount, productSize);
        realMemUsed += ArrayProfile::CalcSize(elemCount, elemSize);
    } while (realMemUsed <= memQuota);
    elemCount -= MIN_BUILD_COUNT;

    return elemCount;
}

int64_t PqflatBuilder::loadSegments(vector<unique_ptr<mercury::MMapFile>> &fileHolder,
                                    vector<ArrayProfile::Pointer> &pkSegs,
                                    vector<ArrayProfile::Pointer> &productSegs,
                                    vector<ArrayProfile::Pointer> &featureSegs) {
    int64_t maxDocNum = 0;
    for (string segmentPath : _segmentList) {
        unique_ptr<MMapFile> pkFile(new MMapFile);
        unique_ptr<MMapFile> productFile(new MMapFile);
        unique_ptr<MMapFile> featureFile(new MMapFile);

        pkFile->open(string(segmentPath + "/" + COMPONENT_PK_PROFILE).c_str(), true);
        productFile->open(string(segmentPath + "/" + COMPONENT_PRODUCT_PROFILE).c_str(), true);
        featureFile->open(string(segmentPath + "/" + COMPONENT_FEATURE_PROFILE).c_str(), true);
        if (!pkFile->isValid() || !productFile->isValid() || !featureFile->isValid()) {
            LOG_ERROR("open segment[%s] failed!", segmentPath.c_str());
            return -1;
        }

        // make segment pk
        ArrayProfile::Pointer segmentPK(new ArrayProfile());
        bool res = segmentPK->load(pkFile->region(), pkFile->region_size());
        pkSegs.emplace_back(segmentPK);
        fileHolder.push_back(move(pkFile));

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

bool PqflatBuilder::writeIndexPackage(size_t maxDocNum,
                                      const string &prefix,
                                      const IndexStorage::Pointer &stg,
                                      const mercury::MMapFile &pkMergeFile,
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

    package.emplace(COMPONENT_PK_PROFILE, pkMergeFile.region(), pkMergeFile.region_size());
    package.emplace(COMPONENT_PRODUCT_PROFILE, productMergeFile.region(), productMergeFile.region_size());
    package.emplace(COMPONENT_FEATURE_PROFILE, featureMergeFile.region(), featureMergeFile.region_size());
    package.emplace(COMPONENT_IDMAP, idMapMergeFile.region(), idMapMergeFile.region_size());

    size_t delMapSize = BitsetHelper::CalcBufferSize(maxDocNum);
    char *delMapData = new char[delMapSize];
    memset(delMapData, 0, delMapSize);
    package.emplace(COMPONENT_DELETEMAP, delMapData, delMapSize);

    string fileName = PQFLAT_INDEX_FILENAME;
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

int PqflatBuilder::BuildIndex(const VectorHolder::Pointer &holder) {
    if (!holder.get()) {
        return IndexError_InvalidArgument;
    }

    if (!_meta.isMatched(*holder)) {
        return mercury::IndexError_UnmatchedMeta;
    }
    // use graph_run_engine
    size_t threadCount = _params.getUint64(PARAM_GENERAL_BUILDER_THREAD_COUNT);

    MuitThreadBatchWorkflow thread_work_engine(threadCount);
    thread_work_engine.Init(nullptr,
                            std::bind(&PqflatBuilder::JobSplit, this, holder));
    thread_work_engine.Run();

    return 0;
}

} // namespace mercury
