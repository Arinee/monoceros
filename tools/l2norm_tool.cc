#include <cstdlib>
#include <fstream>
#include <iostream>
#include <climits>
#include "gflags/gflags.h"
#include "common/vecs_file_holder.h"
#include "common/txt_file_holder.h"
#include "framework/algorithm/mips_reformer.h"
#include "framework/utility/mmap_file.h"
#include "framework/index_framework.h"
#include "framework/vector_holder.h"
#include "common/params_define.h"
#include "utils/index_meta_helper.h"
#include <float.h>
#include "framework/utility/time_helper.h"

using namespace mercury;
using namespace std;

DEFINE_int32(dimension, 64, "data dimension");
DEFINE_string(method, "L2", "method: L2, IP, HAMMING");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, " ", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_string(input_file, "", "input file name");
DEFINE_bool(from_mips, false, "source");
DEFINE_int32(m, 3, "m");

int main(int argc, char *argv[]) {
    // gflags
    gflags::SetUsageMessage("Usage: mips_transform_tool <plugin files' path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    //read meta first
    IndexMeta meta;
    if (!IndexMetaHelper::parseFrom(FLAGS_type, 
                FLAGS_method, 
                FLAGS_dimension, 
                meta)) {
        return -1;
    }
    
    TxtFileHolder::Pointer txtHolder;
    txtHolder.reset(new TxtFileHolder(meta.type(), meta.dimension(), FLAGS_input_first_sep, FLAGS_input_second_sep));
    if (!txtHolder->load(FLAGS_input_file)) {
        cerr << "Load input error: " << FLAGS_input_file << endl;
        return -1;
    }

    assert(txtHolder.get() != nullptr);
    cerr << IndexMetaHelper::toString(meta) << endl;
    std::cout << "Prepare data done!" << std::endl;
    
    auto iter = txtHolder->createIterator();
    if (!iter) {
        LOG_ERROR("Create iterator for holder failed");
        return -1;
    }
        
    ElapsedTime elapsed_time;
    for (; iter->isValid(); ) {
        float l2 = 0;
        for(size_t i = 0; i < meta.dimension(); i++){
            float value = *(reinterpret_cast<float*>((char*)iter->data() + i * 4));
            if(FLAGS_from_mips && (i + FLAGS_m) >= meta.dimension()){
                value = 0.f;
            }
            l2 += (value) * (value);
        }

        std::cout << "id:" << iter->key() << ",l2:" << std::sqrt(l2)  << ", l2 squared:" << l2 <<  endl;
        iter->next();
    }
    /*
    MipsReformer mips(FLAGS_M, FLAGS_U, max);
    Vector<float> *out = new Vector<float>();
    FILE *fp = fopen(FLAGS_output_file.c_str(), "wb");

    if (!fp) {
        LOG_ERROR("Fopen file [%s] with wb failed:[%s]", FLAGS_output_file.c_str(), strerror(errno));
        return -1;
    }   
    
    std::string feature_value;
    char str[8192] = {0};
    iter->reset();
    for(iter->next(); iter->isValid();){
        out->clear();
        feature_value.clear();
        feature_value.reserve(8192);

        mips.transFeature((float*)(iter->data()), 64, out);
        for(auto it_out = out->begin(); it_out != out->end(); ){
            feature_value.append(std::to_string(*it_out));
            if(++it_out != out->end()){
                feature_value.append(",");
            }
        }
        
        size_t len = snprintf(str, 8192, "%ld %s\n", iter->key(), feature_value.c_str());
        size_t cnt = fwrite(str, 1, len, fp);
        if (cnt != len) {
            LOG_ERROR("Write file header to file [%s] fail", FLAGS_output_file.c_str());
            //fclose(fp);
            //return -1;
        }
        iter->next();
    }
    
    fclose(fp);  
    delete out;
    std::cout << "Transform finished!" << std::endl;*/
    return 0;
}
