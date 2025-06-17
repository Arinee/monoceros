#ifndef __MERCURY_CORE_IVF_SEARCHER_H__
#define __MERCURY_CORE_IVF_SEARCHER_H__

#include <memory>
#include <string>
#include <vector>
#include "src/core/framework/index_framework.h"
#include "../general_search_context.h"
#include "../searcher.h"
#include "ivf_index.h"

MERCURY_NAMESPACE_BEGIN(core);

class IvfSearcher : public Searcher
{
public:
    typedef std::shared_ptr<IvfSearcher> Pointer;
public:
    IvfSearcher();
    //! Init from params
    int Init(IndexParams &params) override;

    int LoadIndex(const std::string& path) override;
    int LoadIndex(const void* data, size_t size) override;
    void SetIndex(Index::Pointer index) override;
    void SetBaseDocId(exdocid_t baseDocId) override;
    //! search by query
    int Search(const QueryInfo& query_info, GeneralSearchContext* context = nullptr) override;

public:
    int SearchIvf(std::vector<CoarseIndex<BigBlock>::PostingIterator>& postings, const void* data, size_t size);

private:
    void PushResult(GeneralSearchContext* context, CoarseIndex<BigBlock>::PostingIterator& iter) const;
private:
    IvfIndex::Pointer index_;
};

MERCURY_NAMESPACE_END(core);

#endif // __MERCURY_CORE_IVF_SEARCHER_H__
