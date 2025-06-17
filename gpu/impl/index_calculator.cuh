/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-06-11 19:17 $
 *
 * $Id: index_calculator.h 2018-06-11 19:17 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef IMPL_INDEX_CALCULATOR_H_
#define IMPL_INDEX_CALCULATOR_H_

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/DeviceVector.cuh"

namespace proxima { namespace gpu {

class Calculator
{
public:
    Calculator() {
    }
    virtual ~Calculator() {
    }
    //! Function call
    virtual int dataPrepare(const char *feature, int num, int feature_size,
            DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream) {
        return 0;
    } 
    virtual int distance(cublasHandle_t handle, cudaStream_t stream, 
            const void *lhs, int qnum, const void *rhs, int fnum, 
            void *distance, const int dim) const {
        return 0;
    }
    virtual int sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
            int dx , int fy, int ctile_num, 
            int topk, cudaStream_t stream) {
        return 0;
    }
    virtual int sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
            int qnum, int fnum, int topk, cudaStream_t stream) {
        return 0;
    }
};

template <class T>
class TypeCalculator : public Calculator
{
public:
    TypeCalculator() {
    }
    virtual ~TypeCalculator() {
    }
    //! Function call
    virtual int dataPrepare(const char *feature, int num, int feature_size,
            DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream) {
        return 0;
    } 
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

#endif //IMPL_INDEX_CALCULATOR_H_
