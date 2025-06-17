#include "ivf_seeker.h"
#include "ivf_posting_iterator.h"
#include "centroid_quantizer.h"
#include "coarse_index.h"
#include "general_search_context.h"
#include "index_ivfflat.h"
#include <assert.h>
#include <limits>
#include <algorithm>
#include <vector>

using namespace std;

namespace mercury {

int IvfSeeker::Init(Index *index)
{
    // TODO ivf flat?
    IndexIvfflat *index_ivfflat = dynamic_cast<IndexIvfflat*>(index);
    if (!index_ivfflat) {
        LOG_ERROR("index is not ivf flat");
        return -1;
    }
    if (!index_ivfflat->get_index_meta()) {
        LOG_ERROR("index meta is nullptr");
        return -1;
    }
    index_meta_ = *index_ivfflat->get_index_meta();

    centroid_quantizer_ = index_ivfflat->get_centroid_quantizer();
    if (!centroid_quantizer_) {
        LOG_ERROR("centroid quantizer is nullptr");
        return -1;
    }

    coarse_index_ = index_ivfflat->get_coarse_index();
    if (!coarse_index_) {
        LOG_ERROR("coarse index is nullptr");
        return -1;
    }

    return 0;
}

vector<CoarseIndex::PostingIterator> 
IvfSeeker::Seek(const void *query, size_t bytes, GeneralSearchContext *context)
{
    vector<CoarseIndex::PostingIterator> result;
    if (!context) {
        return result;
    }
    if (_coarseScanRatios.empty()) {
        return result;
    }

    float probe_ratio = _coarseScanRatios.back();
    if (context->updateCoarseParam()) {
        probe_ratio = context->getCoarseProbeRatio();
    }

    size_t leafNum = centroid_quantizer_->get_centroid_resource()->getLeafCentroidNum();
    auto nprobe = (size_t) (probe_ratio * leafNum);
    if (unlikely(nprobe == 0)) {
        LOG_WARN("Nprobe is less than 1, set to 1");
        nprobe = 1;
    }

    vector<size_t> levelScanLimits;
    auto &roughMeta = centroid_quantizer_->get_centroid_resource()->getRoughMeta();
    if (_coarseScanRatios.size() == 2 && roughMeta.levelCnt == 2) {
        float levelScanRatio = _coarseScanRatios[0];
        if (context->updateLevelScanParam()) {
            levelScanRatio = context->getLevelScanRatio();
        }
        //TODO only support 2 level
        auto levelScanNum = (size_t) (levelScanRatio * roughMeta.centroidNums[0]);
        levelScanLimits.push_back(levelScanNum);
    }

    vector<uint32_t>&& label_list = 
        centroid_quantizer_->Search(query, bytes, nprobe, levelScanLimits, &index_meta_);
    for(auto label: label_list){
        result.push_back(coarse_index_->search(label));
    }

    return result;
}


} // namespace mercury

