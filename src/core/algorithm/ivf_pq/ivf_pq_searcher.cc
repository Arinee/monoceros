#include "ivf_pq_searcher.h"
#include "query_distance_matrix.h"
#include "src/core/utils/my_heap.h"
#include "pq_dist_scorer.h"
#include "../query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

IvfPqSearcher::IvfPqSearcher()
{
    if (!index_) {
        index_.reset(new IvfPqIndex());
    }
}

int IvfPqSearcher::Init(IndexParams &params) {
    index_->SetIndexParams(params);
    return 0;
}

int IvfPqSearcher::LoadIndex(const std::string& path) {
    //TODO
    return -1;
}

int IvfPqSearcher::LoadIndex(const void* data, size_t size) {
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }

    return 0;
}

void IvfPqSearcher::SetIndex(Index::Pointer index) {
    index_ = std::dynamic_pointer_cast<IvfPqIndex>(index);
}

void IvfPqSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

void IvfPqSearcher::PushResult(GeneralSearchContext* context, const std::vector<DistNode>& result_vec) const {
    bool with_pk = index_->WithPk();
    for (size_t i = 0; i < result_vec.size(); i++) {
        const DistNode& node = result_vec.at(i);
        docid_t docid = node.key;
        pk_t pk = 0;
        if (unlikely(with_pk)) {
            pk = index_->GetPk(docid);
        }

        docid_t glo_doc_id = docid + base_docid_;
        if (deletion_map_retriever_.isValid() && deletion_map_retriever_(glo_doc_id)) {
            continue;
        }
        context->emplace_back(pk, glo_doc_id, 0);
    }
}

//! search by query
int IvfPqSearcher::Search(const QueryInfo& query_info, GeneralSearchContext* context) {
    // QueryInfo query_info(query_str); // groupinfos 对应的 topks
    // if (!query_info.MakeAsSearcher()) {
    //     LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
    //     return -1;
    // }

    std::vector<CoarseIndex<BigBlock>::PostingIterator> ivf_postings;
    if (index_->SearchIvf(ivf_postings, query_info.GetVector(), query_info.GetVectorLen(), query_info.GetContextParams()) != 0) {
        LOG_ERROR("Failed to call SearchIvf.");
        return -1;
    }

    //初始化PQ距离向量
    QueryDistanceMatrix * qdm = new QueryDistanceMatrix(index_->GetIndexMeta(), &(index_->GetCentroidResource()));
    if (!qdm->initDistanceMatrix(query_info.GetVector())) {
        LOG_ERROR("Init qdm distance matrix error");
        return -1;
    }

    size_t pq_scan_num = index_->GetIndexParams().getUint64(PARAM_PQ_SCAN_NUM);
    MyHeap<DistNode> result(pq_scan_num);
    PqDistScorer scorer(&index_->GetPqCodeProfile());
    for (auto &posting : ivf_postings) {
        docid_t docid = INVALID_DOC_ID;
        while ((docid = posting.next()) != INVALID_DOC_ID) {
            float dist = scorer.score(docid, qdm);
            result.push(DistNode(docid, dist));
        }
    }

    result.sortByDocId();
    PushResult(context, result.getData());

    return 0;
}

MERCURY_NAMESPACE_END(core);
