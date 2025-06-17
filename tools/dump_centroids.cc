#include <cstdlib>
#include <fstream>
#include <iostream>
#include "pq_helper.h"
#include "common/vecs_file_holder.h"
#include "common/txt_file_holder.h"
#include "gflags/gflags.h"
#include "framework/utility/mmap_file.h"
#include "framework/index_framework.h"
#include "framework/vector_holder.h"
#include "common/params_define.h"
#include "utils/index_meta_helper.h"
#include "train/centroid_trainer.h"
#include "framework/utility/time_helper.h"
#include "framework/index_package.h"
#include "index/centroid_resource.h"

DEFINE_string(from, "rough_matrix", "available value: rough_matrix, index");
DEFINE_string(input, "./rough_matrix", "input file");
DEFINE_string(rough_matrix_output, "rough_matrix.txt", "rough matrix output file");

using namespace mercury;
using namespace std;

int main(int argc, char *argv[]) {
    // gflags
    gflags::SetUsageMessage("Usage: dump_centroids <plugin files' path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);


    CentroidResource::Pointer centroidResource;
    centroidResource.reset(new CentroidResource);

    if (FLAGS_from == "rough_matrix") {
        centroidResource.reset(new CentroidResource);
        mercury::MMapFile roughFile, integrateFile;
        roughFile.open(FLAGS_input.c_str(), true);
        int bret = centroidResource->init((void *)roughFile.region(),
                                        roughFile.region_size());
        if (!bret) {
            LOG_ERROR("centroid resource init error");
            return -1;
        }

        ofstream outfile;
        outfile.open(FLAGS_rough_matrix_output);
        if (!outfile) {
            cerr << "open output file error: " << FLAGS_rough_matrix_output << endl;
            return -1;
        }
        outfile << fixed << setprecision(6);

        for (size_t level = 0; level < centroidResource->getRoughMeta().levelCnt; ++level) {
            for(size_t i = 0; i < centroidResource->getRoughMeta().centroidNums[level]; i++){
                outfile << i << ";";
                const float *values = (const float *)centroidResource->getValueInRoughMatrix(level, i);
                size_t dim = centroidResource->getRoughMeta().elemSize / sizeof(float);
                for(size_t debug_i = 0; debug_i < dim; ++debug_i) {
                    if (debug_i + 1 < dim) {
                        outfile << values[debug_i] << ",";
                    } else {
                        outfile << values[debug_i] << endl;

                    }
                }
            }
        }
    } else if (FLAGS_from == "index") {
        IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
        auto file_handle = storage->open(FLAGS_input, false);
        IndexPackage package;
        if (!package.load(file_handle, false)) {
            return false;
        }

        auto *pRoughSegment = package.get(COMPONENT_ROUGH_MATRIX);
        if (!pRoughSegment) {
            LOG_ERROR("get component %s error", COMPONENT_ROUGH_MATRIX.c_str());
            return -1;
        }

        bool bret = centroidResource->init((void *)pRoughSegment->getData(),
                                           pRoughSegment->getDataSize());
        if (!bret) {
            LOG_ERROR("centroid resource init error");
            return -1;
        }


        ofstream outfile;
        outfile.open(FLAGS_rough_matrix_output);
        if (!outfile) {
            cerr << "open output file error: " << FLAGS_rough_matrix_output << endl;
            return -1;
        }
        outfile << fixed << setprecision(6);

        for (size_t level = 0; level < centroidResource->getRoughMeta().levelCnt; ++level) {
            for(size_t i = 0; i < centroidResource->getRoughMeta().centroidNums[level]; i++){
                outfile << i << ";";
                const float *values = (const float *)centroidResource->getValueInRoughMatrix(level, i);
                size_t dim = centroidResource->getRoughMeta().elemSize / sizeof(float);
                for(size_t debug_i = 0; debug_i < dim; ++debug_i) {
                    if (debug_i + 1 < dim) {
                        outfile << values[debug_i] << ",";
                    } else {
                        outfile << values[debug_i] << endl;

                    }
                }
            }
        }
    } else {
        cerr << "Can not recognize from argument :" << endl;
        return -1;
    }


    return 0;
}

