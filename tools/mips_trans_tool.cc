#include <cstdlib>
#include <fstream>
#include <iostream>
#include <climits>
#include "gflags/gflags.h"
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

DEFINE_int32(dimension, 256, "data dimension");
DEFINE_string(method, "L2", "method: L2, IP, HAMMING");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_string(transform_input, "goods_vec.dat", "input file name");
DEFINE_string(output_file, "goods_vec.dat.out", "input file name");
DEFINE_double(U, 0.38, "U");
DEFINE_int32(M, 3, "M");
DEFINE_uint32(report_time, 15000, "report time interval");

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
    if (!txtHolder->load(FLAGS_transform_input)) {
        cerr << "Load input error: " << FLAGS_transform_input << endl;
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
        
    int count = 0;
    ElapsedTime elapsed_time;
    float max = FLT_MIN;
    for (; iter->isValid(); ) {
        //shared_ptr<char> data(new char[elemSize], std::default_delete<char[]>());
        //memcpy(data.get(), reinterpret_cast<const char *>(iter->data()), elemSize);
        //uint64_t key = iter->key();
        float l2 = 0;
        for(size_t i = 0; i < meta.dimension(); i++){
            float *value = reinterpret_cast<float*>((char*)iter->data() + i * 4);
            l2 += (*value) * (*value);
            //cout << "get float:" << *value;
        }
        l2 = sqrt(l2);
        if(l2 > max){
            max = l2;
        }
        count++;
        if(elapsed_time.elapsed() > FLAGS_report_time){
            elapsed_time.update();
            std::cout << "program count:" << count << endl;
        }
        iter->next();
    }
    std::cout << "l2 max:" << max <<  endl;

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
    std::cout << "Transform finished!" << std::endl;
    return 0;
}
