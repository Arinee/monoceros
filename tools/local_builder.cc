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
#include "index/centroid_resource.h"
#include "utils/index_meta_helper.h"
#include "builder/ivfpq_builder.h"
#include "builder/flat_builder.h"
#include "builder/cat_flat_builder.h"
#include "builder/pqflat_builder.h"
#include "builder/ivfflat_builder.h"
#include "builder/cat_ivfflat_builder.h"
#include "builder/cat_ivfpq_builder.h"
#include "framework/utility/time_helper.h"

using namespace mercury;
using namespace std;

DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(builder_class, "IvfpqBuilder", "The register name of builder");
DEFINE_string(index_prefix, "./", "The dir of output indexes");
DEFINE_bool(need_train, false, "need train, default: false");
DEFINE_int32(quota, 10, "quota in GB");
DEFINE_int32(dimension, 256, "data dimension");
DEFINE_string(method, "L2", "method: L2, IP, HAMMING");
DEFINE_bool(need_pqcodebook, false, "need pq codebook");
DEFINE_int32(threads, 16, "threads num, default 16 core");
DEFINE_string(rough_centroid_num, "1000,100", "Train rough centroid number");
DEFINE_int32(integrate_centroid_num, 256, "Train integrate centroid number");
DEFINE_int32(integrate_fragment_num, 32, "Train fragement number");
DEFINE_int32(train_sample_count, 3000000, "Train sample count");
DEFINE_string(build_input, "input.dat", "build input");
DEFINE_string(train_input, "train.dat", "train input");
DEFINE_string(intermediate_path, "./temp", "intermediate path");
DEFINE_string(rough_matrix, "rough_matrix", "rough matrix");
DEFINE_string(integrate_matrix, "integrate_matrix", "integrate matrix");
DEFINE_string(input_file_type, "txt", "input file type:txt, vecs, cat_txt, cat_vecs");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(faiss_index_name, "IDMap,HNSW32", "faiss index name, only support with IDMap, default: IDMap,HNSW32");



VectorHolder::Pointer getHolderFromFile(const IndexMeta &meta, const std::string &inputFile)
{
    VectorHolder::Pointer result;
    if (FLAGS_input_file_type == string("txt")) {
        TxtFileHolder::Pointer txtHolder =
                make_shared<TxtFileHolder>(meta.type(),
                                           meta.dimension(),
                                           FLAGS_input_first_sep,
                                           FLAGS_input_second_sep);
        if (!txtHolder->load(inputFile)) {
            cerr << "Load input error: " << inputFile << endl;
            return result;
        }
        result = txtHolder;
    } else if (FLAGS_input_file_type == string("vecs")) {
        VecsFileHolder::Pointer vecsHolder = make_shared<VecsFileHolder>(meta);
        if (!vecsHolder->load(inputFile)) {
            cerr << "Load input error: " << inputFile << endl;
            return result;
        }
        result = vecsHolder;
    }
    else if (FLAGS_input_file_type == string("cat_vecs")) {
        VecsFileHolder::Pointer vecsHolder = make_shared<VecsFileHolder>(meta, true);
        if (!vecsHolder->load(inputFile)) {
            cerr << "Load input error: " << inputFile << endl;
            return result;
        }
        result = vecsHolder;
    }
    else if (FLAGS_input_file_type == string("cat_txt")) {
        IndexMeta meta;
        if (!IndexMetaHelper::parseFrom(FLAGS_type, FLAGS_method, FLAGS_dimension, meta)) {
            return result;
        }
        TxtFileHolder::Pointer txtHolder =
                make_shared<TxtFileHolder>(meta.type(),
                                           meta.dimension(),
                                           FLAGS_input_first_sep,
                                           FLAGS_input_second_sep, true);
        if (!txtHolder->load(inputFile)) {
            cerr << "Load input error: " << inputFile << endl;
            return result;
        }
        result = txtHolder;
    } else {
        cerr << "Wrong input file type: " << FLAGS_input_file_type << endl;
        return result;
    }
    return result;
}


int main(int argc, char *argv[]) {
    // gflags
    gflags::SetUsageMessage("Usage: knn_build_example <plugin files' path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Load plugins first
    mercury::IndexPluginBroker broker;
    for (int i = 1; i < argc; ++i) {
        const char *file_path = argv[i];

        if (!broker.emplace(file_path)) {
            std::cerr << "Failed to load plugin " << file_path << std::endl;
        } else {
            std::cout << "Loaded plugin " << file_path << std::endl;
        }
    }
    
    /*
    mercury::IndexBuilder::Pointer builder =
        mercury::InstanceFactory::CreateBuilder(FLAGS_builder_class.c_str());
    if (!builder) {
        std::cerr << "Failed to create builder " << FLAGS_builder_class << std::endl;
        return 1;
    } else {
        std::cout << "Created builder" << FLAGS_builder_class << std::endl;
    }
    */

    // Load Index using Storage
    mercury::IndexStorage::Pointer storage =
        mercury::InstanceFactory::CreateStorage(FLAGS_storage_class.c_str());
    if (!storage) {
        std::cerr << "Failed to create storage " << FLAGS_storage_class
                  << std::endl;
        return 3;
    } else {
        std::cout << "Created storage " << FLAGS_storage_class << std::endl;
    }

    IndexMeta meta;
    if (!IndexMetaHelper::parseFrom(FLAGS_type, FLAGS_method, FLAGS_dimension, meta)) {
        cerr << "Failed to parse index meta " << endl;
        return -1;
    }
    cerr << IndexMetaHelper::toString(meta) << endl;

    VectorHolder::Pointer holder = getHolderFromFile(meta, FLAGS_build_input);
    if (!holder) {
        cerr << "Failed to load file holder, file: " << FLAGS_build_input;
        return -1;
    }
    std::cout << "Prepare data done!" << std::endl;
    
    mercury::IndexParams params; 
    params.set(mercury::PARAM_PQ_BUILDER_INTERMEDIATE_PATH, FLAGS_intermediate_path);
    params.set(mercury::PARAM_GENERAL_BUILDER_MEMORY_QUOTA, FLAGS_quota*1024L*1024L*1024L);
    if(FLAGS_builder_class == "CatFlatBuilder")
        params.set(mercury::PARAM_GENERAL_BUILDER_THREAD_COUNT, 1);
    else
        params.set(mercury::PARAM_GENERAL_BUILDER_THREAD_COUNT, FLAGS_threads);
    
    std::cout <<  FLAGS_builder_class << "|" << FLAGS_rough_matrix << "|" << FLAGS_integrate_matrix << endl;
    /*mercury::PQCodebook::Pointer pqCodebook;
    if (!mercury::preparePQCodebook(meta, FLAGS_rough_matrix, FLAGS_integrate_matrix, pqCodebook)) {
        std::cerr << "Failed to preapre pq codebook" << std::endl;
        return -1;
    }*/

    params.set(mercury::PARAM_ROUGH_MATRIX, FLAGS_rough_matrix);
    params.set(mercury::PARAM_INTEGRATE_MATRIX, FLAGS_integrate_matrix);
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_COARSE_CENTROID_NUM, FLAGS_rough_centroid_num);
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_PRODUCT_CENTROID_NUM, FLAGS_integrate_centroid_num);
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_FRAGMENT_NUM, FLAGS_integrate_fragment_num);
    params.set(mercury::PARAM_GENERAL_BUILDER_TRAIN_SAMPLE_COUNT, FLAGS_train_sample_count);
    params.set(mercury::PARAM_FAISS_INDEX_NAME, FLAGS_faiss_index_name);
    
    ElapsedTime elapsedTime;
    if(FLAGS_builder_class == "IvfpqBuilder"){
        IvfpqBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "CatIvfpqBuilder"){
        CatIvfpqBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "FlatBuilder")
    {
        FlatBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "CatFlatBuilder")
    {
        CatFlatBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "CatIVFFlatBuilder")
    {
        CatIvfFlatBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "PqflatBuilder")
    {
        PqflatBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "IVFFlatBuilder")
    {
        IvfflatBuilder builder;
        int ret = builder.Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        ret = builder.BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        ret = builder.DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        builder.Cleanup();
    }
    else if(FLAGS_builder_class == "FaissIndexBuilder")
    {
        ElapsedTime costTime;
        mercury::IndexBuilder::Pointer builder =
            mercury::InstanceFactory::CreateBuilder(FLAGS_builder_class.c_str());
        if (!builder) {
            std::cerr << "Failed to create builder " << FLAGS_builder_class << std::endl;
            return 1;
        } else {
            std::cout << "Created builder" << FLAGS_builder_class << std::endl;
        }
        int ret = builder->Init(meta, params);
        if(ret != 0){
            std::cerr << "Failed to init builder" << std::endl;
            return -1;
        }
        cout << "Init cost: " << costTime.update() << "ms" << endl;

        if (FLAGS_need_train) {

            VectorHolder::Pointer trainHolder = getHolderFromFile(meta, FLAGS_train_input);
            if (!trainHolder) {
                cerr << "Failed to load file holder, file: " << FLAGS_train_input;
                return -1;
            }
            ret = builder->Train(trainHolder);
            if (ret != 0) {
                std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
                return -1;
            }
            std::cout << "Train finished, Cost Time:" << costTime.update() << "ms" << std::endl;
        }

        ret = builder->BuildIndex(holder);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        std::cout << "Build finished, Cost Time:" << costTime.update() << "ms" << std::endl;


        ret = builder->DumpIndex(FLAGS_index_prefix, storage);
        if (ret != 0) {
            std::cerr << "Failed to build in builder, ret=" << ret << std::endl;
            return -1;
        }
        std::cout << "Dump finished, Cost Time:" << costTime.update() << "ms" << std::endl;
    }
    else
    {
        std::cout << "un-support build type error" << std::endl;
        return -1;
    }
    
    std::cout << "Train & Build & Dump finished, Cost Time:" << elapsedTime.elapsed() << "ms" << std::endl;
    return 0;
}
