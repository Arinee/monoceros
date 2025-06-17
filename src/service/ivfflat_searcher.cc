#include "ivfflat_searcher.h"
#include "index/index_ivfflat.h"
#include "utils/my_heap.h"
#include <assert.h>
#include <limits>
#include <algorithm>

using namespace std;

namespace mercury {


IvfflatSearcher::IvfflatSearcher()
    : _segNum(0),
    _indexProvider(nullptr)
{}

IvfflatSearcher::~IvfflatSearcher()
{
    Unload();
    Cleanup();
}

int IvfflatSearcher::Unload()
{
    _segmentFactory.Unload();
    _indexProvider = nullptr;
    return 0;
}

int IvfflatSearcher::Cleanup(void)
{
    return _segmentFactory.Cleanup();
}
    
int IvfflatSearcher::Init(const IndexParams &params)
{
    if (_segmentFactory.Init(params) != 0) {
        return -1;
    }
    return 0;
}

int IvfflatSearcher::Load(IvfFlatIndexProvider *indexProvider)
{
    if (!indexProvider || !indexProvider->get_segment_list().size()) {
        LOG_ERROR("IndexProvider is illegal");
        return -1;
    }
    _indexProvider = indexProvider;

    if (_segmentFactory.Load(_indexProvider) != 0) {
        LOG_ERROR("SegmentSearcherFactory Init error.");
        return -1;
    }

    auto segments = _indexProvider->get_segment_list();
    _indexMeta = *segments[0]->get_index_meta();
    _segNum = segments.size();

    return 0;
}

// TODO custom seeker
int IvfflatSearcher::Search(const void *query, size_t bytes, size_t topk, GeneralSearchContext* context)
{
    MyHeap<DistNode> finalHeap(topk);

    // seek per segment
    for (segid_t segid = 0; segid < _segNum; ++segid) {
        auto segmentSearcher = 
            _segmentFactory.Make(segid, query, bytes, context);
        segmentSearcher->seekAndPush(finalHeap);
    }

    // sort by accurate distance and push to context
    finalHeap.sort();
    for (auto node: finalHeap.getData()) {
        key_t pk = _indexProvider->getPK(node.key);
        context->emplace_back(pk, node.key, node.dist);
    }

    return 0;
}

} // namespace mercury

