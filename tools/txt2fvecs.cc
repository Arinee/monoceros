#include <iostream>
#include <framework/index_framework.h>
#include "gflags/gflags.h"
#include "common/txt_file_holder.h"
#include "utils/index_meta_helper.h"
#include "framework/utility/time_helper.h"

using namespace mercury;
using namespace std;

DEFINE_string(input, "input.txt", "txt input file");
DEFINE_string(input_first_sep, ";", "input first sep");
DEFINE_string(input_second_sep, ",", "input second sep");
DEFINE_string(output, "output.vecs", "vecs output file");
DEFINE_string(type, "float", "available type: float, int16, int8, binary");
DEFINE_string(method, "L2", "available method: L2, IP, HAMMING");
DEFINE_int32(dimension, 256, "data dimension");

bool writeVecsOutput(TxtFileHolder::Pointer holder)
{

    FILE *wfp = fopen(FLAGS_output.c_str(), "wb");
    if (!wfp) {
        cerr << "Open file error. " << FLAGS_output << endl;
        return false;
    }
    
    auto iter = holder->createIterator();
    size_t vectorSize = holder->sizeofElement();
    int dimension = holder->dimension();
    for (; iter->isValid(); iter->next()) {
        size_t ret = 0;
        ret = fwrite(&dimension, sizeof(int), 1, wfp);
        if (ret != 1) {
            cerr << "Write dimension error. " << endl;
            fclose(wfp);
            return false;
        }
        const void * feature = iter->data();
        ret = fwrite(feature, vectorSize, 1, wfp);
        if (ret != 1) {
            cerr << "Write feature error. " << endl;
            fclose(wfp);
            return false;
        }
    }

    fclose(wfp);
    return true;
}

int main(int argc, char * argv[]) 
{
    //gflags
    gflags::SetUsageMessage("Usage: txt2vecs [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    IndexMeta meta;
    if (!IndexMetaHelper::parseFrom(FLAGS_type,
                                    FLAGS_method,
                                    FLAGS_dimension,
                                    meta)) {
        cerr << "Index meta parse error." << endl;
        return -1;
    }
    cout << IndexMetaHelper::toString(meta) << endl;

    ElapsedTime time;

    TxtFileHolder::Pointer txtHolder = 
        make_shared<TxtFileHolder>(meta.type(), 
                meta.dimension(), 
                FLAGS_input_first_sep, 
                FLAGS_input_second_sep);
    if (!txtHolder->load(FLAGS_input)) {
        cerr << "Load input error: " << FLAGS_input << endl;
        return -1;
    }

    bool ret = writeVecsOutput(txtHolder);
    if (!ret) {
        cerr << "write vecs output failed" << endl;
        return -1;
    }

    cout << "Elapsed time: " << time.elapsed() << "ms" << endl;

    return 0;
}

