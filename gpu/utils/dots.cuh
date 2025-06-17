/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-06-01 16:42 $
 *
 * $Id: dots.cuh 2018-06-01 16:42 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef UTILS_DOTS_CUH_
#define UTILS_DOTS_CUH_
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace proxima { namespace gpu {

template <typename T>
__global__ void selfDots(T* data, T* dots, int n, int d) {
    T accumulator = 0;
    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_id < n) {
        for (int i = 0; i < d; i++) {
            T value = data[i + global_id * d];
            accumulator += value * value;
        }
        dots[global_id] = accumulator;
    }    
}


template<typename T>
struct Gemm 
{
public:
    static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            float fAlpha, 
                            const T *A, int lda,
                            const T *B, int ldb,
                            float fBeta,
                            T *C, int ldc) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
};

template<>
struct Gemm<float>
{
    static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            float fAlpha, 
                            const float *A, int lda,
                            const float *B, int ldb,
                            float fBeta,
                            float *C, int ldc) {
        return cublasSgemm(handle, transa, transb, m, n, k,
                       &fAlpha, A, lda, B, ldb, &fBeta, C, ldc);
    }
};

} }
#endif //UTILS_DOTS_CUH_

