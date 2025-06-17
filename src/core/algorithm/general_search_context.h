#ifndef __MERCURY_GENERAL_SEARCH_CONTEXT_H__
#define __MERCURY_GENERAL_SEARCH_CONTEXT_H__
#include <assert.h>
#include <queue>
#include <memory>
#include "pq_common.h"
#include "src/core/common/params_define.h"
#include "src/core/utils/string_util.h"
#include "src/core/framework/index_framework.h"
#include "src/core/framework/attr_retriever.h"
#include "putil/mem_pool/Pool.h"

MERCURY_NAMESPACE_BEGIN(core);
/*! General Search Context
 */

class GeneralSearchContext : public VectorService::SearchContext {
public:
    typedef std::shared_ptr<GeneralSearchContext> Pointer;
    GeneralSearchContext() 
        : _levelScanRatio(0.0f),
        _updateLevelScanParam(false),
        _coarseProbeRatio(0.0f),
        _updateCoarseParam(false),
        _integrateMaxIteration(0),
        _updateIntegrateParam(false),
        _searchMethod(IndexDistance::kMethodUnknown)
    {}

    GeneralSearchContext(const IndexParams &params) 
        : _levelScanRatio(0.0f),
        _updateLevelScanParam(false),
        _coarseProbeRatio(0.0f),
        _updateCoarseParam(false),
        _integrateMaxIteration(0),
        _updateIntegrateParam(false),
        _searchMethod(IndexDistance::kMethodUnknown)
    {
        update(params);
    }

    //! Retrieve search result
    virtual const std::vector<SearchResult> & Result() const override 
    {
        return _result;
    }

    //! Retrieve result object for output
    std::vector<SearchResult> &Result(void)
    {
        return _result;
    }

    void clean(void) 
    {
        _result.clear();
    }

    void push_back(SearchResult &document) 
    {
        _result.push_back(document);
    }

    void emplace_back(key_t pk, gloid_t gloid, score_t score, uint32_t poolId = 0) 
    {
        _result.emplace_back(pk, gloid, score, poolId);
    }

    bool updateLevelScanParam() const 
    {
        return _updateLevelScanParam;
    }

    bool updateCoarseParam() const 
    {
        return _updateCoarseParam;
    }

    bool updateIntegrateParam() const 
    {
        return _updateIntegrateParam;
    }

    float getLevelScanRatio() const 
    {
        return _levelScanRatio;
    }

    float getCoarseProbeRatio() const 
    {
        return _coarseProbeRatio;
    }

    void setCoarseProbeRatio(float ratio_)
    {
        _coarseProbeRatio = ratio_;
    }

    size_t getIntegrateMaxIteration() const
    {
        return _integrateMaxIteration;
    }

    void setIntegrateMaxIteration(size_t num_) {
        _integrateMaxIteration = num_;
    }

    IndexDistance::Methods getSearchMethod() const {
        return _searchMethod;
    }

    void setSearchMethod(IndexDistance::Methods method) {
        _searchMethod = method;
    }

    void setFilter(const CustomFilter &filter) 
    {
        _filter = filter;
    }

    void setFilter(CustomFilter &&filter) 
    {
        _filter = std::forward<CustomFilter>(filter);
    }

    const CustomFilter & getFilter(void) const
    {
        return _filter;
    }

    const AttrRetriever & getAttrRetriever(void) const
    {
        return _attrRetriever;
    }

    void setAttrRetriever(const AttrRetriever& retriever) {
        _attrRetriever = retriever;
    }

    void setFeeder(const CustomFeeder &feeder) 
    {
        _feeder = feeder;
    }

    void setFeeder(CustomFeeder &&feeder) 
    {
        _feeder = std::forward<CustomFeeder>(feeder);
    }

    const CustomFeeder & getFeeder(void) const
    {
        return _feeder;
    }

    void setDocFeeder(const CustomDocFeeder &feeder)
    {
        _docFeeder = feeder;
    }

    void setDocFeeder(CustomDocFeeder &&feeder)
    {
        _docFeeder = std::forward<CustomDocFeeder>(feeder);
    }

    const CustomDocFeeder & getDocFeeder(void) const
    {
        return _docFeeder;
    }

    size_t& getVisitedDocNum() {
        return _visitedDocNum;
    }

    //ivf postings getter
    std::vector<off_t>& getRealSlotIndexs(void)
    {
        return _realSlotIndexs;
    }

    //group ivf postings getter
    std::vector<std::vector<off_t>>& getAllGroupRealSlotIndexs(void)
    {
        return _allGroupRealSlotIndexs;
    }

    void SetSessionPool(putil::mem_pool::Pool* session_pool)
    {
        _sessionPool = session_pool;
    }

    putil::mem_pool::Pool *GetSessionPool()
    {
        return _sessionPool;
    }

private:
    //! Update the context with new parameters
    void update(const IndexParams &params)
    {
        std::string coarseScanRatioStr = params.getString(PARAM_PQ_SEARCHER_COARSE_SCAN_RATIO);
        std::vector<float> coarseScanRatioVec;
        StringUtil::fromString(coarseScanRatioStr, coarseScanRatioVec, ",");
        float levelScanRatio = 0.0f, coarseScanRatio = 0.0f;
        if (coarseScanRatioVec.size() >= 2) {
            levelScanRatio = coarseScanRatioVec[0];
            coarseScanRatio = coarseScanRatioVec[1];
        } else if (coarseScanRatioVec.size() == 1) {
            coarseScanRatio = coarseScanRatioVec[0];
        }
        if (levelScanRatio > 0.0f) {
            _updateLevelScanParam = true;
            _levelScanRatio = levelScanRatio;
        }
        if (coarseScanRatio > 0.0f) {
            _updateCoarseParam = true;
            _coarseProbeRatio = coarseScanRatio;
        }
        size_t integrateMaxIteration = 
            params.getUint64(PARAM_PQ_SEARCHER_PRODUCT_SCAN_NUM);
        if (integrateMaxIteration > 0) {
            _updateIntegrateParam = true;
            _integrateMaxIteration = integrateMaxIteration;
        }

        params.get(PARAM_GENERAL_SEARCHER_SEARCH_METHOD, &_searchMethod);
    }

private:
    std::vector<SearchResult> _result;
    std::vector<off_t> _realSlotIndexs;
    std::vector<std::vector<off_t>> _allGroupRealSlotIndexs;

private:
    float _levelScanRatio;
    bool _updateLevelScanParam;
    float _coarseProbeRatio;
    bool _updateCoarseParam;
    size_t _integrateMaxIteration;
    bool _updateIntegrateParam;
    IndexDistance::Methods _searchMethod;
    CustomFilter _filter;
    AttrRetriever _attrRetriever;
    CustomFeeder _feeder;
    CustomDocFeeder _docFeeder;
    size_t _visitedDocNum = 0;

private:
    putil::mem_pool::Pool *_sessionPool = nullptr;
};

MERCURY_NAMESPACE_END(core);
#endif // __MERCURY_GENERAL_SEARCH_CONTEXT_H__
