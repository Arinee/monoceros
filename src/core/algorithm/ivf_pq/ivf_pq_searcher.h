#ifndef __MERCURY_CORE_IVF_PQ_SEARCHER_H__
#define __MERCURY_CORE_IVF_PQ_SEARCHER_H__

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "../general_search_context.h"
#include "../searcher.h"
#include "ivf_pq_index.h"

MERCURY_NAMESPACE_BEGIN(core);

class IvfPqSearcher : public Searcher
{
public:
    typedef std::shared_ptr<IvfPqSearcher> Pointer;
public:
    IvfPqSearcher();
    //! Init from params
    int Init(IndexParams &params) override;

    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    //! search by query
    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr) override;

private:
    void PushResult(GeneralSearchContext* context, const std::vector<DistNode>& result_vec) const;

private:
    IvfPqIndex::Pointer index_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_IVF_PQ_SEARCHER_H__
