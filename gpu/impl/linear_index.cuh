/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-05-23 11:22 $
 *
 * $Id: linear_index.cuh 2018-05-23 11:22 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef IMPL_LINEAR_INDEX_CUH_
#define IMPL_LINEAR_INDEX_CUH_

#include "../utils/Tensor.cuh"
#include "../utils/GpuResources.h"
#include "../utils/Tensor.cuh"
#include "../utils/DeviceVector.cuh"
#include "index_calculator.cuh"
#include <thrust/device_vector.h>

namespace proxima { namespace gpu {

class LinearIndex
{ 
public:
    LinearIndex(int dim, int feature_size, Calculator *cal, GpuResources *res) :
    _featureDim(dim),
    _featureSize(feature_size),
    _featuresCount(0),
    _device(0),
    _cal(cal),
    _res(res){
    }

    virtual ~LinearIndex() {
    }

    virtual int init(int device); 
    /**
     * @brief add feature to gpu global memory
     * @param feature: the frist addr of feature
     * @param num: the num of our feature(vector) 
     * @return 0:success -1:failed 
     */
    virtual int add(const char * feature, int num);
    /**
     * @param queries: query feature
     * @param qnum: query num
     * @topk: return topk result
     * @outIndices: the index of topk doc
     */
    virtual int search(const void *queries,  float *out_distances,
             int *out_indices, int qnum, int topk);
private:
    int calRowColTileSize(int qnum, int topk, int &ctile_num, int &tile_cols, int &rtile_num, int &tile_rows);
private:
    // Dimensionality of our features 
    const int _featureDim;
    // length of one features 
    const int _featureSize;
    // the count of features
    int _featuresCount;
    // gpu device no
    int _device;
    Calculator *_cal;
    GpuResources *_res;
    DeviceVector<char> _rawData;
    // (x-y)^2 = x^2 + y^2 - 2xy
    // x^2
    DeviceVector<char> _dataDots;
};

} }

#endif //IMPL_LINEAR_INDEX_CUH_
