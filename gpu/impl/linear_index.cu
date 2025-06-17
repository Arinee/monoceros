
#include "linear_index.cuh"
#include "../utils/DeviceDefs.cuh"
#include "../utils/DeviceTensor.cuh"
#include "../utils/Limits.cuh"
#include "../utils/dots.cuh"
#include "selectk.cuh"
#include <cuda_runtime.h>
#include "../utils/CopyUtils.cuh"
#include <cuda.h>

// extern double elapsed ();
namespace proxima { namespace gpu {

int LinearIndex::init(int device) {
    int num_dev = -1;
    CUDA_VERIFY(cudaGetDeviceCount(&num_dev));
    if (device >= num_dev) {
        return -1;
    }
    _device = device;
    return 0;
}

int LinearIndex::add(const char * feature, int num)
{
    // DeviceScope scope(_device);
    auto stream = _res->getDefaultStream(_device);
    _cal->dataPrepare(feature, num, _featureSize, _rawData, _dataDots, stream);
    _featuresCount += num;
    return 0;
}

int LinearIndex::calRowColTileSize(int qnum, int topk, int &ctile_num, int &tile_cols, int &rtile_num, int &tile_rows)
{
    // todo: cal tile row/col num
    ctile_num = 128;
    tile_cols = _featuresCount / ctile_num;
    if (tile_cols < topk) {
        ctile_num = 64;
        tile_cols = _featuresCount / ctile_num;
        if (tile_cols < topk) {
            ctile_num = 1;
            tile_cols = _featuresCount;
        }
    }
    tile_rows = std::min(qnum, 32);
    rtile_num = qnum / tile_rows;
    if (rtile_num <= 0) {
        rtile_num = 1;
        tile_rows = qnum;
    }
    // printf("tileinfo: %d %d %d %d\n", ctile_num, tile_cols, rtile_num, tile_rows);
    return 0;
}

int LinearIndex::search(const void *val, float *out_distances,
             int *out_indices, int qnum, int topk) {
    // double t0 = elapsed();
    if (qnum <= 0) {
        return -1;
    }
    if (topk <= 0 || topk > 1024) {
        return -2;
    }
    if (topk > _featuresCount) {
        topk = _featuresCount;
    }
    DeviceScope scope(_device);

    auto default_stream = _res->getDefaultStreamCurrentDevice();
    auto& mem = _res->getMemoryManagerCurrentDevice();
    auto streams = _res->getAlternateStreamsCurrentDevice();
    auto handle = _res->getBlasHandleCurrentDevice();
    streamWait(streams, {default_stream});

    // cal x*y float or halp type
    DeviceTensor<char, 1, true> distances(mem, {int(qnum * _featuresCount * sizeof(float))}, default_stream);
    char *centroids_view = _rawData.data();
    cublasSetStream(handle, default_stream);
    _cal->distance(handle, default_stream, (const void *)val, qnum, (const void *)centroids_view, _featuresCount, (void *)distances.data(), _featureDim);
    cudaDeviceSynchronize();
    // double t1 = elapsed();
    // printf("dist_cost: %.3f\n", t1-t0);

    // cal tile row/col
    int tile_cols_num = 0;
    int tile_cols = 0;
    int tile_rows_num = 0;
    int tile_rows = 0;
    calRowColTileSize(qnum, topk, tile_cols_num, tile_cols, tile_rows_num, tile_rows);

    // top k tile 
    DeviceTensor<float, 1, true> tile_topk_distance(mem, {qnum * tile_cols_num * topk}, default_stream);
    DeviceTensor<int, 1, true> tile_topk_indices(mem, {qnum * tile_cols_num * topk}, default_stream);
    float *tile_topk_distance_view = tile_topk_distance.data();
    int *tile_topk_indices_view = tile_topk_indices.data();
    int cur_stream = 0;
    _cal->sortBlock((void*)distances.data(), (void *)_dataDots.data(), tile_topk_distance_view, tile_topk_indices_view, 
                       qnum, _featuresCount, tile_cols_num, topk, streams[cur_stream]);
    cudaDeviceSynchronize();
    // double t2 = elapsed();
    // printf("topk_cost1: %.3f\n", t2-t1);

    // top k merge
    _cal->sortMerge(tile_topk_distance_view, tile_topk_indices_view,
                        out_distances, out_indices, 
                       qnum, tile_cols_num * topk, topk, streams[cur_stream]);
    cudaDeviceSynchronize();

    // double t3 = elapsed();
    // printf("topk_cost2: %.3f\n", t3-t2);
    return topk;
}

} }

