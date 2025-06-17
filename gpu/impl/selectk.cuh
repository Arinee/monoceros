/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-05-25 11:47 $
 *
 * $Id: selectk.cuh 2018-05-25 11:47 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef IMPL_SELECTK_CUH_
#define IMPL_SELECTK_CUH_
#include "../utils/Select.cuh"
#include "../utils/StaticUtils.h"

namespace proxima { namespace gpu {

template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void selectMinK(T *distances,  
                           T *dots,
                           float *outDistances,
                           int *outIndices,
                           int qx, int fy, int k, T initK) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];
    int q_idx = blockIdx.x;
    int f_idy = blockIdx.y;
    int tile_disance_step = fy / gridDim.y;
    int tile_disance_num = tile_disance_step;
    if (blockIdx.y + 1 == gridDim.y) {
        tile_disance_num = fy - blockIdx.y * tile_disance_step;
    }

    BlockSelect<float, int, false, Comparator<float>, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(initK, -1, smemK, smemV, k);
    int limit = utils::roundDown(tile_disance_num, kWarpSize);
    int didx = q_idx * fy + f_idy * tile_disance_step; 
    int fidx = f_idy * tile_disance_step;
    int i = threadIdx.x;
    if (dots != NULL) {
        for (; i < limit; i += blockDim.x) {
            float v = Math<T>::add(dots[fidx + i], distances[didx + i]);
            heap.add(v, fidx + i);
        }
        if (i < tile_disance_num) {
            float  v = Math<T>::add(dots[fidx + i], distances[didx + i]);
            heap.addThreadQ(v, fidx + i);
        }
    } else {
        for (; i < limit; i += blockDim.x) {
            float v = Math<T>::add(distances[didx + i], 0);
            heap.add(v, fidx + i);
        }
        if (i < tile_disance_num) {
            float  v =Math<T>::add(distances[didx + i], 0);
            heap.addThreadQ(v, fidx + i);
        }
    }
    heap.reduce();
    
    int out_idx = q_idx * gridDim.y * k + blockIdx.y * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
       outDistances[out_idx + i] = smemK[i];
       outIndices[out_idx + i] = smemV[i];
    }
}

template <typename T, int NumWarpQ, int NumThreadQ, int ThreadsPerBlock>
__global__ void selectMinKKV(T* distances, 
                             int * indices,
                             T* outDistances,
                             int* outIndices,
                             int qnum, int fnum, int k, T initK) {
    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;
  
    __shared__ T smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];
  
    BlockSelect<T, int, false, Comparator<T>, NumWarpQ, NumThreadQ, ThreadsPerBlock> heap(initK, -1, smemK, smemV, k);
  
    int row = blockIdx.x;
    if (row >= qnum) {
        return;
    }

    int idx = row * fnum;
    int i = threadIdx.x;
    int limit = utils::roundDown(fnum, kWarpSize);
    for (; i < limit; i += blockDim.x) {
       T v = distances[idx + i];
       heap.add(v, indices[idx + i]);
    }
    if (i < fnum) {
       T v = distances[idx + i];
       heap.addThreadQ(v, indices[idx + i]);
    }
    heap.reduce();
    
    int topk_idx = row * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
       outDistances[topk_idx + i] = smemK[i];
       outIndices[topk_idx + i] = smemV[i];
    }
}

} }
#endif //IMPL_SELECTK_CUH_

