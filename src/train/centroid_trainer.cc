#include <fstream>
#include <assert.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <dirent.h>
#include <chrono>
#include <thread>
#include "centroid_trainer.h"
#include "utils/string_util.h"
#include "common/common_define.h"
#include "index/pq_index_error.h"
#include "cluster/kmedoids_cluster.h"
#include "cluster/multistage_cluster.h"
#include "framework/utility/thread_pool.h"
#include "framework/utility/buffer_sampler.h"
#include "framework/utility/bitset_helper.h"
#include "framework/utility/thread_pool.h"

using namespace std;
using namespace mercury;

bool CentroidTrainer::Init(const mercury::IndexMeta &meta, const mercury::IndexParams &params)
{
    _meta = meta;
    _params = params;
    
    size_t threadCount = _params.getUint64(PARAM_GENERAL_BUILDER_THREAD_COUNT);
    if (threadCount > 0) {
        _pool.reset(new mercury::ThreadPool(false, threadCount));
    } else {
        _pool.reset(new mercury::ThreadPool);
    }

    return true;
}

int CentroidTrainer::trainIndexImpl(const mercury::VectorHolder::Pointer &holder)
{
    int ret = 0;
    if(_roughOnly) {
        LOG_INFO("Train Rough Only");
        ret = trainRough(holder);
        if (ret != 0) {
            LOG_ERROR("fail to train rough centroids");
            return ret;
        }
    } else {
        ret = trainRoughAndIntegrate(holder);
        if (ret != 0) {
            LOG_ERROR("fail to train rough and integrate centroids");
            return ret;
        }
    }
    return ret;

}


int CentroidTrainer::trainRoughAndIntegrate(const mercury::VectorHolder::Pointer &holder)
{
    if (!holder.get()) {
        return IndexError_InvalidArgument;
    }
    if (!_meta.isMatched(*holder)) {
        return mercury::IndexError_UnmatchedMeta;
    }

    size_t dimension = holder->dimension();
    uint32_t sampleCount = _params.getUint32(PARAM_GENERAL_BUILDER_TRAIN_SAMPLE_COUNT);
    string roughCentroidNum = _params.getString(PARAM_PQ_BUILDER_TRAIN_COARSE_CENTROID_NUM);
    vector<uint32_t> roughCentroidNumVec;
    StringUtil::fromString(roughCentroidNum, roughCentroidNumVec, ",");
    uint32_t integrateCentroidNum = _params.getUint32(PARAM_PQ_BUILDER_TRAIN_PRODUCT_CENTROID_NUM);
    uint32_t fragmentNum = _params.getUint32(PARAM_PQ_BUILDER_TRAIN_FRAGMENT_NUM);
    size_t roughLevelCount = roughCentroidNumVec.size();
    if (!(sampleCount > 0 && dimension > 0 && roughLevelCount > 0 && integrateCentroidNum > 0 && fragmentNum > 0)) {
        LOG_ERROR("param error, sample count[%u], dimension[%zd], rough level count[%zd], integrate centroid[%u], fragment[%u]", 
                sampleCount, dimension, roughLevelCount, integrateCentroidNum, fragmentNum);
        return IndexError_InvalidArgument;
    }
    if (dimension % fragmentNum != 0) {
        LOG_ERROR("dimension[%zd] can not be divided by fragment number[%u]", 
                dimension, fragmentNum);
        return IndexError_InvalidArgument;
    }
    size_t roughElemSize = _meta.sizeofElement();
    size_t integrateElemSize = roughElemSize / fragmentNum;

    CentroidResource::RoughMeta roughMeta(roughElemSize, roughLevelCount, roughCentroidNumVec);
    CentroidResource::IntegrateMeta integrateMeta(integrateElemSize, fragmentNum, integrateCentroidNum);
    _resource.reset(new CentroidResource);
    if (!_resource->create(roughMeta, integrateMeta)) {
        LOG_ERROR("failed to create centroid resource.");
        return IndexError_CentroidResource;
    }

    // prepare cluster feature data
    mercury::BufferSampler sampler(sampleCount, roughElemSize);

    auto iter = holder->createIterator();
    if (!iter) {
        LOG_ERROR("Create iterator for holder failed");
        return IndexError_Holder;
    }
    for (; iter->isValid(); iter->next()) {
        string data(reinterpret_cast<const char *>(iter->data()), roughElemSize);
        sampler.fill(data.data());
    }

    auto featureList = sampler.split();
    LOG_DEBUG("train data read finished, sample count:%ld, rough level:%ld", featureList.size(), roughLevelCount);

    MultistageCluster cluster;

    // rough cluster
    cluster.mount(featureList.data(), featureList.size(), _meta.type(), roughElemSize);

    if (roughLevelCount == 1) {
        KmedoidsCluster stageCluster1;
        stageCluster1.setClusterCount(roughCentroidNumVec[0]);
        stageCluster1.setDistance(_meta.measure());
        stageCluster1.setOnlyMeans(true);
        cluster.cluster(*_pool, stageCluster1);
    } else if (roughLevelCount == 2) {
        KmedoidsCluster stageCluster1;
        stageCluster1.setClusterCount(roughCentroidNumVec[0]);
        stageCluster1.setDistance(_meta.measure());
        stageCluster1.setOnlyMeans(true);
        KmedoidsCluster stageCluster2;
        stageCluster2.setClusterCount(roughCentroidNumVec[1]);
        stageCluster2.setDistance(_meta.measure());
        cluster.cluster(*_pool, stageCluster1, stageCluster2);
    } else {
        LOG_ERROR("not support cluster with level[%zd]", roughLevelCount);
        return IndexError_Clustering;
    }

    size_t centroidIndex = 0;
    std::vector<size_t> fs;
    for (const auto &it : cluster.getCentroids()) {
        if (it.follows() > 0 && _sanityCheck) fs.push_back(it.follows());
        _resource->setValueInRoughMatrix(0, centroidIndex, it.feature());
        if (roughLevelCount == 2) {
            size_t leafIndex = 0;
            for (const auto &jt : it.subitems()) {
                _resource->setValueInRoughMatrix(1, 
                        centroidIndex * roughCentroidNumVec[1] + leafIndex++, 
                        jt.feature());
            }
        }
        centroidIndex++;
    }

    if (_sanityCheck) {
        std::sort(fs.begin(), fs.end(),std::greater<size_t>());
        std::cout << "------------------- size of each cluster ----------------" << std::endl;
        for (const auto& e : fs) std::cout << e << ",";
        std::cout << std::endl;

        //check whether this clustering is ill-formed
        auto validCentroidNum = fs.size();
        auto expectedCentroidNum = cluster.getCentroids().size();
        std::cout << "expectedCentroidNum: " << expectedCentroidNum
            << ",   validCentroidNum: " << validCentroidNum << std::endl;
        if (validCentroidNum < expectedCentroidNum * _sanityCheckCentroidNumRatio) {
            std::cerr << "ERROR: clustering is ill-formed." << std::endl;
            return -1;
        }
    }

    LOG_DEBUG("rough cluster finished! Valid centroid num: %zu", centroidIndex);

    // integrate cluster
    KmedoidsCluster fragmentCluster;
    fragmentCluster.setClusterCount(integrateCentroidNum);
    fragmentCluster.setDistance(_meta.measure());
    fragmentCluster.setOnlyMeans(true);
    for (size_t f = 0; f < fragmentNum; ++f) {
        vector<ClusterFeatureAny> fragFeatureList;
        fragFeatureList.reserve(featureList.size());
        for (auto fullFeature : featureList) {
            fragFeatureList.push_back(reinterpret_cast<const char*>(fullFeature) + f * integrateElemSize);
        }
        cluster.mount(fragFeatureList.data(), fragFeatureList.size(), _meta.type(), integrateElemSize);
        if (!cluster.cluster(*_pool, fragmentCluster)) {
            LOG_ERROR("integrate cluster error in fragment[%zd]!", f);
            return IndexError_Clustering;
        }

        // add to integrate resource
        centroidIndex = 0;
        for (const auto &it : cluster.getCentroids()) {
            _resource->setValueInIntegrateMatrix(f, centroidIndex++, it.feature());
        }
        LOG_DEBUG("integrate cluster in fragment[%zd] finished!", f);
    }

    _isTrainDone = true;

    //flush to file

    string rough_file = _params.getString(PARAM_ROUGH_MATRIX);
    string integrate_file = _params.getString(PARAM_INTEGRATE_MATRIX);
    bool res = _resource->DumpToFile(rough_file, integrate_file);
    if (!res) {
        return IndexError_DumpPackageIndex;
    }

    return 0;
}

int CentroidTrainer::trainRough(const mercury::VectorHolder::Pointer &holder)
{
    if (!holder.get()) {
        return IndexError_InvalidArgument;
    }
    if (!_meta.isMatched(*holder)) {
        return mercury::IndexError_UnmatchedMeta;
    }

    size_t dimension = holder->dimension();
    uint32_t sampleCount = _params.getUint32(PARAM_GENERAL_BUILDER_TRAIN_SAMPLE_COUNT);
    string roughCentroidNum = _params.getString(PARAM_PQ_BUILDER_TRAIN_COARSE_CENTROID_NUM);
    vector<uint32_t> roughCentroidNumVec;
    StringUtil::fromString(roughCentroidNum, roughCentroidNumVec, ",");

    size_t roughLevelCount = roughCentroidNumVec.size();
    if (!(sampleCount > 0 && dimension > 0 && roughLevelCount > 0)) {
        LOG_ERROR("param error, sample count[%u], dimension[%zd], rough level count[%zd]",
                  sampleCount, dimension, roughLevelCount);
        return IndexError_InvalidArgument;
    }
    size_t roughElemSize = _meta.sizeofElement();

    CentroidResource::RoughMeta roughMeta((uint32_t)roughElemSize,
                                          (uint32_t)roughLevelCount,
                                          roughCentroidNumVec);
    _resource.reset(new CentroidResource);
    if (!_resource->create(roughMeta)) {
        LOG_ERROR("failed to create centroid resource.");
        return IndexError_CentroidResource;
    }

    // prepare cluster feature data
    mercury::BufferSampler sampler(sampleCount, roughElemSize);

    auto iter = holder->createIterator();
    if (!iter) {
        LOG_ERROR("Create iterator for holder failed");
        return IndexError_Holder;
    }
    for (; iter->isValid(); iter->next()) {
        string data(reinterpret_cast<const char *>(iter->data()), roughElemSize);
        sampler.fill(data.data());
    }

    auto featureList = sampler.split();
    LOG_DEBUG("train data read finished, sample count:%ld, rough level:%ld", featureList.size(), roughLevelCount);

    MultistageCluster cluster;

    // rough cluster
    cluster.mount(featureList.data(), featureList.size(), _meta.type(), roughElemSize);

    if (roughLevelCount == 1) {
        KmedoidsCluster stageCluster1;
        stageCluster1.setClusterCount(roughCentroidNumVec[0]);
        stageCluster1.setDistance(_meta.measure());
        stageCluster1.setOnlyMeans(true);
        cluster.cluster(*_pool, stageCluster1);
    } else if (roughLevelCount == 2) {
        KmedoidsCluster stageCluster1;
        stageCluster1.setClusterCount(roughCentroidNumVec[0]);
        stageCluster1.setDistance(_meta.measure());
        stageCluster1.setOnlyMeans(true);
        KmedoidsCluster stageCluster2;
        stageCluster2.setClusterCount(roughCentroidNumVec[1]);
        stageCluster2.setDistance(_meta.measure());
        cluster.cluster(*_pool, stageCluster1, stageCluster2);
    } else {
        LOG_ERROR("not support cluster with level[%zd]", roughLevelCount);
        return IndexError_Clustering;
    }

    size_t centroidIndex = 0;
    std::vector<size_t> fs;
    for (const auto &it : cluster.getCentroids()) {
        if (it.follows() > 0 && _sanityCheck) fs.push_back(it.follows());
        _resource->setValueInRoughMatrix(0, centroidIndex, it.feature());
        if (roughLevelCount == 2) {
            size_t leafIndex = 0;
            for (const auto &jt : it.subitems()) {
                _resource->setValueInRoughMatrix(1,
                                                 centroidIndex * roughCentroidNumVec[1] + leafIndex++,
                                                 jt.feature());
            }
        }
        centroidIndex++;
    }

    if (_sanityCheck) {
        std::sort(fs.begin(), fs.end(),std::greater<size_t>());
        std::cout << "------------------- size of each cluster ----------------" << std::endl;
        for (const auto& e : fs) std::cout << e << ",";
        std::cout << std::endl;

        //check whether this clustering is ill-formed
        auto validCentroidNum = fs.size();
        auto expectedCentroidNum = cluster.getCentroids().size();
        std::cout << "expectedCentroidNum: " << expectedCentroidNum
                  << ",   validCentroidNum: " << validCentroidNum << std::endl;
        if (validCentroidNum < expectedCentroidNum * 0.5 ||
            fs.front() > featureList.size() * 0.1) {
            std::cerr << "ERROR: clustering is ill-formed." << std::endl;
            return -1;
        }
    }

    LOG_DEBUG("rough cluster finished! Valid centroid num: %zu", centroidIndex);
    _isTrainDone = true;

    //flush to file
    string rough_file = _params.getString(PARAM_ROUGH_MATRIX);
    bool res = _resource->DumpToFile(rough_file);
    if (!res) {
        return IndexError_DumpPackageIndex;
    }

    return 0;
}
