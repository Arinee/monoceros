#include <iostream>
#include <stdlib.h>
#include <gflags/gflags.h>
#include <iomanip>
#include "framework/utility/mmap_file.h"

using namespace mercury;
using namespace std;

DEFINE_string(input, "top100.gt", "ground truth file");
DEFINE_bool(only_key, true, "only output key");

// format: 
// topk (key, score) (ke, score) ...
// int, uint64_t, float, uint64_t, float, ...
void printContent(const void *region, size_t regionSize)
{
    int topk = *(const int *)region;
    size_t recordBytes = sizeof(int) + topk * (sizeof(uint64_t) + sizeof(float));
    size_t num = regionSize / recordBytes;
    fprintf(stdout, "num: %lu, topk: %d, recordBytes: %lu\n", num, topk, recordBytes);
    for (size_t i=0; i < num; ++i) {
        const char *recordBase = ((const char *)region + sizeof(int) + i * recordBytes);
        for (int k = 0; k < topk; ++k) {
            cout<<setiosflags(ios::fixed);
            cout<<setprecision(6);
            cout << *(const uint64_t *)(recordBase + k * (sizeof(uint64_t) + sizeof(float)));
            if (!FLAGS_only_key) {
                cout << "("
                    << *(const float *)(recordBase + sizeof(uint64_t) + k * (sizeof(uint64_t) + sizeof(float)))
                    << ")";
            }
            if (k != topk - 1) {
                cout << " ";
            }
        }
        cout << endl;
    }
}

int main(int argc, char * argv[]) 
{
    //gflags
    gflags::SetUsageMessage("Usage: print_groundtruth [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    MMapFile mmapFile;
    if (!mmapFile.open(FLAGS_input.c_str(), true)) {
        std::cerr << "Open file error: " << FLAGS_input << std::endl;
        return -1;
    }
    printContent(mmapFile.region(), mmapFile.region_size());

    return 0;
}
