#include "exhaustive_searcher.h"
#include "index/index.h"
#include "utils/my_heap.h"
#include <assert.h>
#include <limits>
#include <algorithm>

using namespace std;

namespace mercury {

ExhaustiveSearcher::ExhaustiveSearcher()
    :_segNum(0),
    _indexProvider(nullptr)
{}

ExhaustiveSearcher::~ExhaustiveSearcher()
{
    Unload();
    Cleanup();
}

int ExhaustiveSearcher::Unload()
{
    _indexProvider = nullptr;
    return 0;
}

int ExhaustiveSearcher::Cleanup(void)
{
    return 0;
}
    
int ExhaustiveSearcher::Init(const IndexParams& /*params*/)
{
    return 0;
}

int ExhaustiveSearcher::Load(BaseIndexProvider *indexProvider)
{
    if (!indexProvider) {
        LOG_ERROR("index provider is null");
        return -1;
    }
    _indexProvider = indexProvider;

    if (_indexProvider->get_segment_list().size() == 0) {
        LOG_ERROR("No segment in index provider");
        return -1;
    }

    auto segments = _indexProvider->get_segment_list();
    for (size_t segid = 0; segid < segments.size(); ++segid) {
        _segNum ++;
        Index* index = segments[segid].get();
        if (!index) {
            return -1;
        }

        OrigDistScorer::Factory scorerFactory;
        if (scorerFactory.Init(index) != 0) {
            LOG_ERROR("OrigDistScorer::Factory Init error.");
            return -1;
        }
        _scorerFactories.push_back(scorerFactory);
    }
    return 0;
}

/* 
 * validate argument in caller
 * context with empty result
 */
int ExhaustiveSearcher::Search(const void* query, size_t /*bytes*/, size_t topk, GeneralSearchContext* context)
{
    MyHeap<DistNode> finalHeap(topk);
    auto segments = _indexProvider->get_segment_list();
    for (segid_t segid = 0; segid < _segNum; ++segid) {
        size_t docNumInSeg = segments[segid]->get_doc_num();
        OrigDistScorer&& 
            scorer = _scorerFactories[segid].Create(context);
        for (docid_t docid = 0; docid < docNumInSeg; ++docid) {
            float dist = scorer.Score(docid, query);
            gloid_t gloid = GET_GLOID(segid, docid);
            finalHeap.push(DistNode(gloid, dist));
        }
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

