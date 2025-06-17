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
#include "train/centroid_trainer.h"
#include "framework/utility/time_helper.h"

using namespace mercury;
using namespace std;

DEFINE_string(index_prefix, "./", "The dir of output indexes");
DEFINE_bool(need_train, true, "need train");
DEFINE_int32(quota, 10, "quota in GB");
DEFINE_int32(dimension, 256, "data dimension");
DEFINE_string(method, "L2", "method: L2, IP, HAMMING");
DEFINE_bool(need_pqcodebook, false, "need pq codebook");
DEFINE_int32(threads, 16, "threads num, default 16 core");
DEFINE_string(rough_centroid_num, "1024", "Train rough centroid number");
DEFINE_bool(rough_only, false, "coarse cluster only");
DEFINE_bool(sanity_check, false, "check whether clustering is valid");
DEFINE_int32(retry_time, 0, "how many retry times if clustering failed");
DEFINE_double(sanity_check_centroid_num_ratio, 0.5, "threshold");
DEFINE_int32(integrate_centroid_num, 256, "Train integrate centroid number");
DEFINE_int32(integrate_fragment_num, 32, "Train fragement number");
DEFINE_int32(train_sample_count, 3000000, "Train sample count");
DEFINE_string(train_input, "train.dat", "train input");
DEFINE_string(intermediate_path, "./temp", "intermediate path");
DEFINE_string(rough_matrix, "rough_matrix", "rough matrix");
DEFINE_string(integrate_matrix, "integrate_matrix", "integrate matrix");
DEFINE_string(input_file_type, "txt", "input file type:txt, vecs, cat_txt, cat_vecs");
DEFINE_string(input_first_sep, " ", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");

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

    mercury::IndexMeta meta; 
    if (!IndexMetaHelper::parseFrom(FLAGS_type, FLAGS_method, FLAGS_dimension, meta)) {
        return -1;
    }
    mercury::VectorHolder::Pointer trainHolder;
    if (FLAGS_input_file_type == string("txt")) {
        //read meta first
        
        TxtFileHolder::Pointer txtHolder;
        txtHolder.reset(new TxtFileHolder(meta.type(), meta.dimension(), FLAGS_input_first_sep, FLAGS_input_second_sep));
        if (!txtHolder->load(FLAGS_train_input)) {
            cerr << "Load input error: " << FLAGS_train_input << endl;
            return -1;
        }
        cout << txtHolder->sizeofElement() << endl;
        trainHolder = txtHolder;
    }
    else if (FLAGS_input_file_type == string("cat_txt")) {
        TxtFileHolder::Pointer txtHolder =
            make_shared<TxtFileHolder>(meta.type(),
                    meta.dimension(),
                    FLAGS_input_first_sep,
                    FLAGS_input_second_sep, true);
        if (!txtHolder->load(FLAGS_train_input)) {
            cerr << "Load input error: " << FLAGS_train_input << endl;
            return -1;
        }
        cout << txtHolder->sizeofElement() << endl;
        trainHolder = txtHolder;
    } else if (FLAGS_input_file_type == string("vecs")) {
        VecsFileHolder::Pointer vecsHolder = make_shared<VecsFileHolder>(meta);
        if (!vecsHolder->load(FLAGS_train_input)) {
            cerr << "Load input error: " << FLAGS_train_input << endl;
            return -1;
        }
        trainHolder = vecsHolder;
        meta = vecsHolder->indexMeta();
    } else if (FLAGS_input_file_type == string("cat_vecs")) {
        VecsFileHolder::Pointer vecsHolder = make_shared<VecsFileHolder>(meta, true);
        if (!vecsHolder->load(FLAGS_train_input)) {
            cerr << "Load input error: " << FLAGS_train_input << endl;
            return -1;
        }
        trainHolder = vecsHolder;
        meta = vecsHolder->indexMeta();
    } else {
        cerr << "Wrong input file type: " << FLAGS_input_file_type << endl;
        return -1;
    }
    assert(trainHolder.get() != nullptr);
    cerr << IndexMetaHelper::toString(meta) << endl;
    std::cout << "Prepare data done!" << std::endl;
    
    mercury::IndexParams params; 
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_COARSE_CENTROID_NUM, FLAGS_rough_centroid_num);
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_PRODUCT_CENTROID_NUM, FLAGS_integrate_centroid_num);
    params.set(mercury::PARAM_PQ_BUILDER_TRAIN_FRAGMENT_NUM, FLAGS_integrate_fragment_num);
    params.set(mercury::PARAM_GENERAL_BUILDER_TRAIN_SAMPLE_COUNT, FLAGS_train_sample_count);
    params.set(mercury::PARAM_ROUGH_MATRIX, FLAGS_rough_matrix);
    params.set(mercury::PARAM_INTEGRATE_MATRIX, FLAGS_integrate_matrix);

    CentroidTrainer centroid_trainer(FLAGS_rough_only, FLAGS_sanity_check,
            FLAGS_sanity_check_centroid_num_ratio);
    bool res = centroid_trainer.Init(meta, params);
    if(!res){
        std::cerr << "Failed to init builder" << std::endl;
        return -1;
    }

    ElapsedTime elapsed_time;
    int currentRetry = 0;
    int ret = 0;
    while(true) {
        ret = centroid_trainer.trainIndexImpl(trainHolder);
        if (ret != 0 && currentRetry++ < FLAGS_retry_time) continue;
        break;
    }
    if (ret != 0) {
        std::cerr << "Failed to train in builder, ret=" << ret << std::endl;
        return ret;
    }
    std::cout << "Train finished, Cost time:" << elapsed_time.elapsed() << std::endl;
    return 0;
}
