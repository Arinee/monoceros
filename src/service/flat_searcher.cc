#include "flat_searcher.h"
#include "index/index.h"
#include "utils/my_heap.h"
#include <assert.h>
#include <limits>
#include <algorithm>

using namespace std;

namespace mercury {

FlatSearcher::FlatSearcher()
    : _segNum(0),
    _indexProvider(nullptr)
{}

FlatSearcher::~FlatSearcher()
{
    Unload();
    Cleanup();
}

int FlatSearcher::Unload()
{
    _indexProvider = nullptr;
    return _segSearcherFactory.Unload();
}

int FlatSearcher::Cleanup(void)
{
    return _segSearcherFactory.Cleanup();
}
    
int FlatSearcher::Init(const IndexParams & /* params */)
{
    return 0;
}

int FlatSearcher::Load(BaseIndexProvider *indexProvider)
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
    _segNum = segments.size();

    return 0;
}

int FlatSearcher::Search(const void *query, size_t bytes, size_t topk, GeneralSearchContext* context)
{
    MyHeap<DistNode> finalHeap(topk);

    // seek per segment
    for (segid_t segid = 0; segid < _segNum; ++segid) {
        auto segmentSearcher = 
            _segSearcherFactory.Make(segid, query, bytes, context);
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

