/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-05-22 14:34 $
 *
 * $Id: gpu_knn_linear.h 2018-05-22 14:34lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef GPU_GPU_KNN_LINEAR_H_
#define GPU_GPU_KNN_LINEAR_H_
#include "impl/linear_index.cuh"
#include "utils/StandardGpuResources.h"
#include "aitheta/index_framework.h"
#include "impl/calculator_factory.h"

namespace proxima { namespace gpu {

/*! Knn Linear Searcher Context
 */
class GpuKnnLinearSearcherContext : public aitheta::IndexSearcher::Context
{
public:
    //! Retrieve search result
    virtual const std::vector<aitheta::IndexSearcher::Document> &
    result(void) const
    {
        return _result;
    }

    //! Retrieve result object for output
    std::vector<aitheta::IndexSearcher::Document> &result(void)
    {
        return _result;
    }

private:
    std::vector<aitheta::IndexSearcher::Document> _result;
};

class GpuKnnLinearSearcher : public aitheta::IndexSearcher
{ 
public:
    GpuKnnLinearSearcher(void);
    virtual ~GpuKnnLinearSearcher() {
        release();
    }
    //! KNN Search in Local
protected:
    void release();
    void
    knnSearchInLocal(size_t topk, const void *query, int qnum,
                     const aitheta::IndexSearcher::Filter &filter,
                     std::vector<aitheta::IndexSearcher::Document> *result);

    //! Initialize Searcher
    virtual int initImpl(const aitheta::SearcherParams &params);
    //! Cleanup Searcher
    virtual int cleanupImpl(void);
    //! Load index from file path or dir
    virtual int loadIndexImpl(const std::string &prefix,
                              const aitheta::IndexStorage::Pointer &stg);
    //! Unload index
    virtual int unloadIndexImpl(void);
    //! KNN Search
    virtual int knnSearchImpl(size_t topk, const void *val, size_t len,
                              Context::Pointer &context);

    //! Create a searcher context
    virtual Context::Pointer
    createContextImpl(const aitheta::SearcherContextParams &) {
        return Context::Pointer(new GpuKnnLinearSearcherContext());
    }
    //! Test if the index is loaded
    bool isLoaded(void) const {
        return (_feature_keys != nullptr && _cal != nullptr && _index != nullptr);
    }
    //! Members
    const void *_features;
    const uint64_t *_feature_keys;
    size_t _feature_size;
    size_t _features_count;
    bool _fast_load;
    std::string _file_name;
    aitheta::IndexStorage::Handler::Pointer _handle;
    int _device_no;
    proxima::gpu::StandardGpuResources _res;
    Calculator *_cal;
    LinearIndex *_index;
};

} }
#endif //GPU_GPU_KNN_LINEAR_H_
