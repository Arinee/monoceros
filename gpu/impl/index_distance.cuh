/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-06-04 20:27 $
 *
 * $Id: index_distance.h 2018-06-04 20:27 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef IMPL_INDEX_DISTANCE_H_
#define IMPL_INDEX_DISTANCE_H_
#include "index_calculator.cuh"

namespace proxima { namespace gpu {

class GPUFloatSquaredEuclideanCalculator : public TypeCalculator<float>
{
public:
    //! Function call
    virtual int dataPrepare(const char *feature, int num, int feature_size,
            DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream); 
    virtual int distance(cublasHandle_t handle, cudaStream_t stream, 
            const void *lhs, int qnum, const void *rhs, int fnum, 
            void *distance, const int dim) const;
    virtual int sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
            int dx , int fy, int ctile_num, 
            int topk, cudaStream_t stream);
    virtual int sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
            int qnum, int fnum, int topk, cudaStream_t stream);
};

class GPUFloatInnerProductCalculator : public TypeCalculator<float>
{
public:
    //! Function call
    virtual int dataPrepare(const char *feature, int num, int feature_size,
            DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream); 
    virtual int distance(cublasHandle_t handle, cudaStream_t stream, 
            const void *lhs, int qnum, const void *rhs, int fnum, 
            void *distance, const int dim) const;
    virtual int sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
            int dx , int fy, int ctile_num, 
            int topk, cudaStream_t stream);
    virtual int sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
            int qnum, int fnum, int topk, cudaStream_t stream);
};

} }
#endif //IMPL_INDEX_DISTANCE_H_

