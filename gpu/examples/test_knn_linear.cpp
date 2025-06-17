
#include "../utils/StandardGpuResources.h"
#include "../utils/CopyUtils.cuh"
#include "../impl/linear_index.cuh"
#include "../impl/index_distance.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    // return  tv.tv_sec + tv.tv_usec * 1e-6;
    return  tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


int main (int argc, char *argv[]) 
{
    if (argc < 5) {
        fprintf(stderr, "usage: feature_num dim qnum topk\n");
        return -1;
    }
    int nb = atoi(argv[1]);
    int d = atoi(argv[2]);
    int qnum = atoi(argv[3]);
    int topk = atoi(argv[4]);

    proxima::gpu::StandardGpuResources resources;
    proxima::gpu::Calculator *cal = new proxima::gpu::GPUFloatSquaredEuclideanCalculator();
    proxima::gpu::LinearIndex linear_index(d, d * sizeof(float), cal, &resources);
    linear_index.init(0);
    std::vector <float> database (nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        // float v = (i/d) % 10;
        // database[i] = v;
        database[i] = drand48();
    }
    int rand[30] = {1,    10,    20,    30,   40,   50,   60,   70,   80,   100,  
                    1000, 10000, 10001, 10002,20000, 30000, 40000, 50000, 60000, 
                    100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 910000, 999999};
    for (int i = 0; i< 30 ; i++) {
        int idx = rand[i] * d;
        for (int j = 0; j < d; j++) {
            database[idx + j] = 2;
        }
    }
    linear_index.add((const char *)database.data(), nb);
    auto stream = resources.getDefaultStream(0);

    printf("search begin, fn:%d dim:%d\n", nb, d);
    for (int i = 0; i < 1000; ++i) {
        std::vector <float> query(d * qnum);
        for (size_t j = 0; j < d * qnum; j++) {
            query[j] = 2;// drand48();
        }
        double t0 = elapsed();
        if (topk > nb) {
            topk = nb;
        }
        std::vector<int> labels(qnum * topk);
        std::vector<float> distances(qnum * topk);
        auto query_view = proxima::gpu::toDevice<char, 1>(&resources, 0, (char *)query.data(), stream, {qnum * d * int(sizeof(float))}); 
        auto outDistances = proxima::gpu::toDevice<char, 1>(&resources, 0, (char *)distances.data(), stream, {qnum * topk * int(sizeof(float))});
        auto outIndices = proxima::gpu::toDevice<int, 1>(&resources, 0, labels.data(), stream, {qnum * topk});
        int ret = linear_index.search((const void *)query_view.data(), (float *)outDistances.data(), outIndices.data(), qnum, topk);
        if (ret < 0) {
            continue;
        }

        proxima::gpu::fromDevice<float>((float *)outDistances.data(), (float *)distances.data(), qnum * topk, stream);
        proxima::gpu::fromDevice<int, 1>(outIndices, (int* )labels.data(), stream);
        printf("cost: %.3f\n", elapsed() - t0);
        /*
        for (int i = 0; i < qnum; ++i) {
            int idx = i * topk;
            for (int j = 0; j < topk; j++) {
                printf("query: %d label: %d dist: %f\n", i, labels[idx + j], distances[idx + j]);
            }
        }
        */
    }
    delete cal;
    cal = NULL;
    return 0;
}

