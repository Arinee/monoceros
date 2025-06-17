#include "pqflat_searcher.h"
#include "index/index_ivfpq.h"
#include "utils/my_heap.h"
#include <assert.h>
#include <limits>
#include <algorithm>

using namespace std;

namespace mercury {

PqflatSearcher::PqflatSearcher()
    : _defaultPqScanNum(PQ_SCAN_NUM),
    _segNum(0),
    _indexProvider(nullptr)
{}

PqflatSearcher::~PqflatSearcher()
{
    Unload();
    Cleanup();
}

int PqflatSearcher::Unload()
{
    _indexProvider = nullptr;
    return _segSearcherFactory.Unload();
}

int PqflatSearcher::Cleanup(void)
{
    return _segSearcherFactory.Cleanup();
}
    
int PqflatSearcher::Init(const IndexParams &params)
{
    params.get(PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM, &_defaultPqScanNum);
    LOG_INFO("Init pqflat searcher, default pqScanNum: %lu", _defaultPqScanNum);
    return 0;
}

int PqflatSearcher::Load(BaseIndexProvider *indexProvider)
{
    if (!indexProvider || !indexProvider->get_segment_list().size()) {
        LOG_ERROR("IndexProvider is illegal");
        return -1;
    }
    _indexProvider = indexProvider;

    if (_segSearcherFactory.Load(_indexProvider) != 0) {
        LOG_ERROR("SegmentSearcherFactory Init error.");
        return -1;
    }

    auto segments = _indexProvider->get_segment_list();
    _indexMeta = *segments[0]->get_index_meta();
    _centroidResource = dynamic_cast<IndexPqflat *>(segments[0].get())->getCentroidResource();
    _segNum = segments.size();

    if (_refinerFactory.Init(_indexProvider) != 0) {
        LOG_ERROR("RefinerFactory Init error.");
        return -1;
    }

    return 0;
}

int PqflatSearcher::Search(const void *query, size_t bytes, size_t topk, GeneralSearchContext* context)
{
    // init heap
    size_t pq_scan_num = _defaultPqScanNum;
    if (context->updateIntegrateParam()) {
        pq_scan_num = context->getIntegrateMaxIteration();
    }
    MyHeap<DistNode> pqHeap(pq_scan_num);

    // init qdm
    QueryDistanceMatrix qdm(_indexMeta, _centroidResource);
    if (!qdm.initDistanceMatrix(query)) {
        LOG_ERROR("Init qdm distance matrix error");
        return -1;
    }

    // seek per segment
    for (segid_t segid = 0; segid < _segNum; ++segid) {
        auto segmentSearcher = 
            _segSearcherFactory.Make(segid, query, bytes, context);
        segmentSearcher->seekAndPush(&qdm, pqHeap);
    }

    MyHeap<DistNode> finalHeap(topk);

    DistanceRefiner &&refiner = _refinerFactory.Create(context);
    refiner.ScoreAndPush(query, pqHeap, finalHeap);

    // sort by accurate distance and push to context
    finalHeap.sort();
    for (auto node: finalHeap.getData()) {
        key_t pk = _indexProvider->getPK(node.key);
        context->emplace_back(pk, node.key, node.dist);
    }
    return 0;
}

} // namespace mercury

