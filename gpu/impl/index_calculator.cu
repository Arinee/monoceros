
#include "selectk.cuh"
#include "../utils/dots.cuh"
#include "index_calculator.cuh"

namespace proxima { namespace gpu {

template <class T>
int TypeCalculator<T>::distance(cublasHandle_t handle,cudaStream_t stream,  
        const void *queries, int qnum, const void *features, int fnum, 
        void *distance, const int dim) const
{
        float alpha = -2.0;
        float beta = 0.0;
        float * val = (float *)queries;
        float * centroids_view = (float *)features;
        float *distance_view = (float *)distance;
        // cal: q * d
        Gemm<T>::gemm(handle, 
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    fnum, qnum, dim, alpha,
                    (float*)centroids_view,
                    dim,
                    (float *)val,
                    dim,
                    beta,
                    distance_view, fnum);
        return 0;
}

template <class T>
int TypeCalculator<T>::sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
        int dx , int fy, int ctile_num, int topk, cudaStream_t stream) {
    constexpr int kThreadsPerBlock = 128;
    auto grid_topk = dim3(dx, ctile_num);
    auto block_topk = dim3(kThreadsPerBlock);
    if (topk <= 32) {
        selectMinK<float, 32, 2, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    } else if (topk <= 64) {
        selectMinK<float, 64, 3, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    } else if (topk <= 128) {
        selectMinK<float, 128, 3, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    } else if (topk <= 256) {
        selectMinK<float, 256, 4, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    } else if (topk <= 512) {
        selectMinK<float, 512, 8, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    } else if (topk <= 1024) {
        selectMinK<float, 1024, 8, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>> 
            ((float *)distance, (float *)dots, (float *)out_distances,                                                    
            out_indices, dx, fy, topk, Limits<float>::getMax()); 
    }
    return 0;
}

template <class T>
int TypeCalculator<T>::sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
        int qnum, int fnum, int topk, cudaStream_t stream) {
    constexpr int kThreadsPerBlock = 128;
    auto grid_topk = dim3(qnum);
    auto block_topk = dim3(kThreadsPerBlock);

    if (topk <= 32) {
        selectMinKKV<float, 32, 2, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    } else if (topk <= 64) {
        selectMinKKV<float, 64, 3, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    } else if (topk <= 128) {
        selectMinKKV<float, 128, 3, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    } else if (topk <= 256) {
        selectMinKKV<float, 256, 4, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    } else if (topk <= 512) {
        selectMinKKV<float, 512, 8, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    } else if (topk <= 1024) {
        selectMinKKV<float, 1024, 8, kThreadsPerBlock><<<grid_topk, block_topk, 0, stream>>>               
             ((float *)distances, indices, 
              (float *)out_distances, out_indices, qnum, fnum, topk, Limits<float>::getMax()); 
    }
    return 0;
}

} }

