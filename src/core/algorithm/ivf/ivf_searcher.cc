#include "ivf_searcher.h"
#include "../query_info.h"

MERCURY_NAMESPACE_BEGIN(core);

IvfSearcher::IvfSearcher()
{
    if (!index_) {
        index_.reset(new IvfIndex());
    }
}

int IvfSearcher::Init(IndexParams &params) {
    index_->SetIndexParams(params);
    return 0;
}

int IvfSearcher::LoadIndex(const std::string& path) {
    //TODO
    return -1;
}

int IvfSearcher::LoadIndex(const void* data, size_t size) {
    if (index_->Load(data, size) != 0) {
        LOG_ERROR("Failed to load index.");
        return -1;
    }

    return 0;
}

void IvfSearcher::SetIndex(Index::Pointer index)
{
    index_ = std::dynamic_pointer_cast<IvfIndex>(index);
}

void IvfSearcher::SetBaseDocId(exdocid_t baseDocId)
{
    base_docid_ = baseDocId;
    index_->SetBaseDocid(baseDocId);
}

void IvfSearcher::PushResult(GeneralSearchContext* context, CoarseIndex<BigBlock>::PostingIterator& iter) const {
    bool with_pk = index_->WithPk();
    while (!iter.finish()) {
        docid_t docid = iter.next();
        pk_t pk = 0;
        if (likely(with_pk)) {
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
int IvfSearcher::Search(const QueryInfo& query_info, GeneralSearchContext* context) {
    // QueryInfo query_info(query_str); // groupinfos 对应的 topks
    // if (index_->GetIndexMeta().type() == IndexMeta::FeatureTypes::kTypeBinary) {
    //     query_info.SetFeatureTypes(IndexMeta::FeatureTypes::kTypeBinary);
    // }
    // if (!query_info.MakeAsSearcher()) {
    //     LOG_ERROR("resolve query failed. query str:%s", query_info.GetRawQuery().c_str());
    //     return -1;
    // }

    std::vector<CoarseIndex<BigBlock>::PostingIterator> ivf_postings;
    if (index_->SearchIvf(ivf_postings, query_info.GetVector(), query_info.GetVectorLen()) != 0) {
        LOG_ERROR("Failed to call SearchIvf.");
        return -1;
    }

    for (auto& ivf_posting : ivf_postings) {
        PushResult(context, ivf_posting);
    }

    return 0;
};

MERCURY_NAMESPACE_END(core);
