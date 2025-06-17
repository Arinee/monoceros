#include "centroid_quantizer.h"
#include "utils/my_heap.h"
#include "framework/search_result.h"

using namespace std;

namespace mercury {

/// load data
void CentroidQuantizer::set_centroid_resource(CentroidResource::Pointer centroid_resource)
{
    centroid_resource_ = centroid_resource;
}

CentroidResource* CentroidQuantizer::get_centroid_resource()
{
    return centroid_resource_.get();
}

vector<uint32_t> CentroidQuantizer::Search(const void *query, size_t /*len*/, size_t nprobe, 
        const vector<size_t>& levelScanLimit, IndexMeta* index_meta)
{
    vector<uint32_t> coarseIndexLabels;
    coarseIndexLabels.reserve(nprobe);

    auto roughMeta = centroid_resource_->getRoughMeta();

    priority_queue<CentroidInfo, vector<CentroidInfo>, std::greater<CentroidInfo>> _centroids;

    auto &centroidNums = roughMeta.centroidNums;
    if (centroidNums.empty()) {
        LOG_WARN("centroidNums is empty");
        return coarseIndexLabels;
    }
    const size_t firstLevel = 0;
    for (uint32_t i = 0; i < centroidNums[firstLevel]; ++i)
    {
        const void* centroidValue = centroid_resource_->getValueInRoughMatrix(firstLevel, i);
        float score = index_meta->distance(query, centroidValue);
        _centroids.emplace(i, score);
    }

    for (size_t level = 1; level < roughMeta.levelCnt; ++level)
    {
        uint32_t centroidNum = roughMeta.centroidNums[level];
        decltype(_centroids) candidate;
        candidate.swap(_centroids);

        size_t scanNum = levelScanLimit[level - 1];
        while (!candidate.empty() && scanNum-- > 0) {
            auto doc = candidate.top();
            candidate.pop();
            for (uint32_t i = 0; i < centroidNum; ++i) {
                uint32_t centroid = doc.index * centroidNum + i;
                const void* centroidValue = centroid_resource_->getValueInRoughMatrix(level, centroid); 
                float dist = index_meta->distance(query, centroidValue);
                _centroids.emplace(centroid, dist);
            }
        }
    }

    while(!_centroids.empty() && coarseIndexLabels.size() < nprobe)
    {
        coarseIndexLabels.push_back(_centroids.top().index);
        _centroids.pop();
    }

    return coarseIndexLabels;
}

bool CentroidQuantizer::DumpLevelOneQuantizer(IndexPackage &package)
{
    //dump centroid resource
    roughMatrix.clear();
    integrateMatrix.clear();
    centroid_resource_->dumpRoughMatrix(roughMatrix);
    centroid_resource_->dumpIntegrateMatrix(integrateMatrix);

    package.emplace(COMPONENT_ROUGH_MATRIX, roughMatrix.data(), roughMatrix.size());
    package.emplace(COMPONENT_INTEGRATE_MATRIX, integrateMatrix.data(), integrateMatrix.size());

    return true;
}

bool CentroidQuantizer::CreateLevelOneQuantizer(map<string, size_t>& stab)
{
    if(!centroid_resource_->IsInit()){
        LOG_ERROR("centroid not initialize");
        return false;
    }

    centroid_resource_->dumpRoughMatrix(roughMatrix);
    centroid_resource_->dumpIntegrateMatrix(integrateMatrix);

    stab.emplace(COMPONENT_ROUGH_MATRIX, roughMatrix.size());
    stab.emplace(COMPONENT_INTEGRATE_MATRIX, integrateMatrix.size());

    return true;
}

bool CentroidQuantizer::LoadLevelOneQuantizer(IndexPackage &package)
{
    centroid_resource_.reset(new CentroidResource);

    auto *pRoughSegment = package.get(COMPONENT_ROUGH_MATRIX);
    if (!pRoughSegment) {
        LOG_ERROR("get component %s error", COMPONENT_ROUGH_MATRIX.c_str());
        return false;
    }
    auto *pIntegrateSegment = package.get(COMPONENT_INTEGRATE_MATRIX);
    if (!pIntegrateSegment) {
        LOG_ERROR("get component %s error", COMPONENT_INTEGRATE_MATRIX.c_str());
        return false;
    }


    bool bret = centroid_resource_->init((void *)pRoughSegment->getData(),
            pRoughSegment->getDataSize(),
            (void *)pIntegrateSegment->getData(), 
            pIntegrateSegment->getDataSize());
    if (!bret) {
        LOG_ERROR("centroid resource init error");
        return false;
    }

    return true;
}

int32_t CentroidQuantizer::CalcLabel(const void *val, size_t len, IndexMeta* index_meta,
        const vector<size_t>& roughLevelScanLimit)
{
    int32_t label = 0;
    vector<uint32_t>&& label_vec = Search(val, len, 1, roughLevelScanLimit, index_meta);
    if(likely(label_vec.size())){
        label = label_vec[0];
    }
    return label;
}

size_t CentroidQuantizer::get_slot_num()
{
    return centroid_resource_->getLeafCentroidNum();
}

}
