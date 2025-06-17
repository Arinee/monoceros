#ifndef __MERCURY_SEARCHER_H__
#define __MERCURY_SEARCHER_H__

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "src/core/framework/deletion_map_retriever.h"
#include "src/core/algorithm/query_info.h"
#include "general_search_context.h"
#include "index.h"

MERCURY_NAMESPACE_BEGIN(core);

class Searcher
{
public:
    typedef std::shared_ptr<Searcher> Pointer;
public:
    Searcher() : base_docid_(0) {};
    //~Searcher();
    //! Init from params
    virtual int Init(IndexParams &params) = 0;

    virtual int LoadDiskIndex(const std::string& disk_index_path, const std::string& medoids_data_path) {
        return -1;
    }
    
    virtual bool Add(docid_t doc_id, const void* data) {
        return true;
    };
    virtual int LoadIndex(const std::string& path) = 0;
    virtual int LoadIndex(const void* data, size_t size) = 0;
    virtual void SetIndex(Index::Pointer index) = 0;
    virtual void SetBaseDocId(exdocid_t baseDocId) = 0;
    virtual IndexMeta::FeatureTypes getFType() {
        return IndexMeta::kTypeUnknown;
    }
    virtual void SetDeletionMapRetriever(const DeletionMapRetriever& retriever) {
        deletion_map_retriever_ = retriever;
    }
    virtual void SetVectorRetriever(const AttrRetriever& retriever) {
        vector_retriever_ = retriever;
    }
    //! search by query
    // 类目层级:c1#topk,c2#topk;类目层级:c3#topk||v1 v2 v3...
    // 1:111#500,112#500;2:222#100||0.1 0.3 0.2
    virtual int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr) = 0;

protected:
    exdocid_t base_docid_;
    DeletionMapRetriever deletion_map_retriever_;
    AttrRetriever vector_retriever_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_SEARCHER_H__
