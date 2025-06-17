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

using namespace mercury;
using namespace std;

DEFINE_int32(dimension, 256, "data dimension");
DEFINE_string(method, "L2", "method: L2, IP, HAMMING");
DEFINE_string(rough_matrix, "rough_matrix", "rough matrix");
DEFINE_string(integrate_matrix, "integrate_matrix", "integrate matrix");
DEFINE_string(index_file, "index", "index file");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");

int main(int argc, char *argv[]) {
    // gflags
    gflags::SetUsageMessage("Usage: knn_build_example <plugin files' path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
 
    IndexStorage::Pointer storage = InstanceFactory::CreateStorage("MMapFileStorage");
    auto file_handle = storage->open(FLAGS_index_file, false);
    IndexPackage package;
    if (!package.load(file_handle, false)) {
        return false;
    }

    CentroidResource::Pointer centroid_resource_;
    centroid_resource_.reset(new CentroidResource);

    auto *pRoughSegment = package.get(COMPONENT_ROUGH_MATRIX);
    if (!pRoughSegment) {
        LOG_ERROR("get component %s error", COMPONENT_ROUGH_MATRIX.c_str());
        return -1;
    }
    auto *pIntegrateSegment = package.get(COMPONENT_INTEGRATE_MATRIX);
    if (!pIntegrateSegment) {
        LOG_ERROR("get component %s error", COMPONENT_INTEGRATE_MATRIX.c_str());
        return -1;
    }

    bool bret = centroid_resource_->init((void *)pRoughSegment->getData(),
            pRoughSegment->getDataSize(),
            (void *)pIntegrateSegment->getData(), 
            pIntegrateSegment->getDataSize());
    if (!bret) {
        LOG_ERROR("centroid resource init error");
        return -1;
    }

    // print 
    for(size_t i = 0; i < centroid_resource_->getLeafCentroidNum(); i++){
        printf("centroid index:%ld, value:", i);
        for(size_t debug_i = 0; debug_i < centroid_resource_->getRoughMeta().elemSize; debug_i+=4){
            float* debug = (float *)(((char *)centroid_resource_->getValueInRoughMatrix(0, i)) + debug_i);
            printf("%f,", *debug);
        }
        printf("\n");
    }
   /*
    centroid_resource_.reset(new CentroidResource);
    mercury::MMapFile roughFile, integrateFile;
    roughFile.open("rough_matrix.recall", true);
    integrateFile.open("integrate_matrix", true);

    bret = centroid_resource_->init((void *)roughFile.region(),
            roughFile.region_size(),
            (void *)integrateFile.region(), 
            integrateFile.region_size());
    if (!bret) {
        LOG_ERROR("centroid resource init error");
        return false;
    }
    
    // print 
    for(size_t i = 0; i < centroid_resource_->getLeafCentroidNum(); i++){
        printf("centroid index:%ld, value:", i);
        for(size_t debug_i = 0; debug_i < centroid_resource_->getRoughMeta().elemSize; debug_i+=4){
            float* debug = ((float *)centroid_resource_->getValueInRoughMatrix(0, i)) + debug_i;
            printf("%f,", *debug);
        }
        printf("\n");
    }
    */
    return 0;
}

