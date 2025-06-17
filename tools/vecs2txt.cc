#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include "gflags/gflags.h"
#include "common/vecs_reader.h"
#include "utils/index_meta_helper.h"

using namespace mercury;
using namespace std;

DEFINE_string(input, "input.vecs", "input file");
DEFINE_string(output, "output.txt", "output file");
DEFINE_string(output_first_sep, ";", "output first sep");
DEFINE_string(output_second_sep, ",", "output second sep");
DEFINE_string(type, "float", "available type: float, int16, int8");
DEFINE_string(method, "L2", "available method: L2, IP, HAMMING");
DEFINE_int32(dimension, 256, "data dimension");

template <typename T>
void printContent(const VecsReader &vecsReader, const string &output)
{
    ofstream outfile;
    outfile.open(output);
    if (!outfile) {
        cerr << "open output file error: " << output << endl;
        return;
    }
    outfile << fixed << setprecision(6);

    size_t dimension = FLAGS_dimension;
    size_t numVecs = vecsReader.numVecs();
    for (size_t i=0; i<numVecs; ++i) {
        outfile << vecsReader.getKey(i) << FLAGS_output_first_sep;
        const T *values = static_cast<const T *>(vecsReader.getVector(i));
        for (size_t k = 0; k < dimension; ++k) {
            outfile << values[k] ;
            if (k != dimension - 1) {
                outfile << FLAGS_output_second_sep;
            }
        }
        outfile << endl;
    }
}

int main(int argc, char * argv[]) 
{
    //gflags
    gflags::SetUsageMessage("Usage: vecs2txt [options]");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    IndexMeta meta;
    if (!IndexMetaHelper::parseFrom(FLAGS_type, 
                FLAGS_method, 
                FLAGS_dimension, 
                meta)) {
        return -1;
    }
    cerr << IndexMetaHelper::toString(meta) << endl;

    VecsReader vecsReader(meta.sizeofElement());
    if (!vecsReader.load(FLAGS_input)) {
        cerr << "Failed to load input: " << argv[1] << endl;
        return -1;
    }

    if (meta.type() == IndexMeta::kTypeFloat) {
        printContent<float>(vecsReader, FLAGS_output);
    } else if (meta.type() == IndexMeta::kTypeDouble) {
        printContent<double>(vecsReader, FLAGS_output);
    } else if (meta.type() == IndexMeta::kTypeInt16) {
        printContent<int16_t>(vecsReader, FLAGS_output);
    } else if (meta.type() == IndexMeta::kTypeInt8) {
        printContent<int8_t>(vecsReader, FLAGS_output);
    } else {
        cerr << "Can not recognize IndexMeta type." 
            << endl;
        exit(-1);
    }
    return 0;
}
