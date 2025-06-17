
#include "index_distance.cuh"
#include "selectk.cuh"
#include "../utils/dots.cuh"
#include "index_calculator.cu"

namespace proxima { namespace gpu {
int GPUFloatSquaredEuclideanCalculator::distance(cublasHandle_t handle,cudaStream_t stream,  
        const void *queries, int qnum, const void *features, int fnum, 
        void *distance, const int dim) const
{
    int ret = TypeCalculator<float>::distance(handle, stream, queries, qnum, features, fnum, distance, dim);
    return ret;
}

int GPUFloatSquaredEuclideanCalculator::dataPrepare(const char *feature, int num, int feature_size,
            DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream)
{
    // todo: gpu mem alloc failed
    device_feature.append((char*)feature, (size_t)num * feature_size, stream, true);
    int features_count = device_feature.size() / feature_size;
    // todo: gpu mem alloc failed
    device_feature_dots.resize(features_count * sizeof(float), stream);
    auto grid = dim3((features_count - 1)/256 + 1);
    auto block = dim3(256);
    selfDots<float><<<grid, block, 0, stream>>>((float *)(device_feature.data()), 
             (float *)(device_feature_dots.data()), features_count, feature_size / sizeof(float));
    return 0;
}

int GPUFloatSquaredEuclideanCalculator::sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
        int dx , int fy, int ctile_num, 
        int topk, cudaStream_t stream)
{
    int ret = TypeCalculator<float>::sortBlock(distance, dots, out_distances, out_indices, dx, fy,ctile_num, topk, stream);
    return ret;
}

int GPUFloatSquaredEuclideanCalculator::sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
        int qnum, int fnum, int topk, cudaStream_t stream)
{
    int ret = TypeCalculator<float>::sortMerge(distances, indices, out_distances, out_indices, qnum, fnum, topk, stream);
    return ret;
}

int GPUFloatInnerProductCalculator::dataPrepare(const char *feature, int num, int feature_size,
    DeviceVector<char> &device_feature, DeviceVector<char> &device_feature_dots, cudaStream_t stream) 
{
    device_feature.append((char*)feature, (size_t)num * feature_size, stream, true);
    return 0;
}

int GPUFloatInnerProductCalculator::distance(cublasHandle_t handle,cudaStream_t stream,  
        const void *queries, int qnum, const void *features, int fnum, 
        void *distance, const int dim) const
{
    float alpha = -1.0;
    float beta = 0.0;
    float * val = (float *)queries;
    float * centroids_view = (float *)features;
    float *distance_view = (float *)distance;
    // cal: q * d
    Gemm<float>::gemm(handle, 
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


int GPUFloatInnerProductCalculator::sortBlock(const void *distance, const void *dots, float *out_distances, int *out_indices, 
        int dx , int fy, int ctile_num, 
        int topk, cudaStream_t stream)
{
    int ret = TypeCalculator<float>::sortBlock(distance, NULL, out_distances, out_indices, dx, fy,ctile_num, topk, stream);
    return ret;
}

int GPUFloatInnerProductCalculator::sortMerge(float *distances, int *indices, float *out_distances, int *out_indices, 
        int qnum, int fnum, int topk, cudaStream_t stream)
{
    int ret = TypeCalculator<float>::sortMerge(distances, indices, out_distances, out_indices, qnum, fnum, topk, stream);
    return ret;
}

} }

