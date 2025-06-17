#include "index_ivf.h"

using namespace std;
using namespace mercury;

bool IndexIvf::CreateIVFIndexFromPackage(map<string, size_t>& stab){
    _centroidQuantizer->CreateLevelOneQuantizer(stab);
    return true;
} 

bool IndexIvf::LoadIVFIndexFromPackage(IndexPackage &package)
{
    if(!_centroidQuantizer->LoadLevelOneQuantizer(package)){
        LOG_ERROR("L1 Quantizer load error");
        return false;
    }

    return true;
}

bool IndexIvf::DumpIVFIndexToPackage(IndexPackage &package, bool /*only_dump_meta*/)
{
    _centroidQuantizer->DumpLevelOneQuantizer(package);
    return true;
}

CentroidQuantizer* IndexIvf::get_centroid_quantizer()
{
    return this->_centroidQuantizer;
}
