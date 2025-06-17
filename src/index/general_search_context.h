#ifndef __MERCURY_GENERAL_SEARCH_CONTEXT_H__
#define __MERCURY_GENERAL_SEARCH_CONTEXT_H__
#include <assert.h>
#include <queue>
#include <memory>
#include "pq_common.h"
#include "common/params_define.h"
#include "utils/string_util.h"
#include "framework/index_framework.h"

namespace mercury {

/*! General Search Context
 */

class GeneralSearchContext : public VectorService::SearchContext {
public:
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

    void emplace_back(key_t pk, gloid_t gloid, score_t score) 
    {
        _result.emplace_back(pk, gloid, score);
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
    float _levelScanRatio;
    bool _updateLevelScanParam;
    float _coarseProbeRatio;
    bool _updateCoarseParam;
    size_t _integrateMaxIteration;
    bool _updateIntegrateParam;
    IndexDistance::Methods _searchMethod;
    CustomFilter _filter;
    CustomFeeder _feeder;
    CustomDocFeeder _docFeeder;
    size_t _visitedDocNum = 0;
};

}; // namespace mercury

#endif // __MERCURY_GENERAL_SEARCH_CONTEXT_H__
