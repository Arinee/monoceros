#include <cstdlib>
#include <fstream>
#include <iostream>
#include "gflags/gflags.h"
#include "index/index_flat.h"
#include "index/index_ivfflat.h"
#include "index/index_ivfpq.h"
#include "index/index_pqflat.h"
#include "common/common_define.h"
#include "framework/index_framework.h"
#include "framework/utility/time_helper.h"

using namespace mercury;
using namespace std;

DEFINE_string(storage_class, "MMapFileStorage", "The register name of storage");
DEFINE_string(index_type, "IvfpqBuilder", "The register name of builder");
DEFINE_string(index_prefix, "./", "The dir of output indexes");

template<class T>
void DoLoad(T& index, IndexStorage::Handler::Pointer& file_handle)
{
    if(file_handle == nullptr){
        std::cerr << "file handle nullptr error!" << std::endl;
        return;
    }

    bool res = index.Load(move(file_handle));
    if(!res){
        std::cerr << "Failed to load index!" << std::endl;
        return;
    }

    cout << "index summary:" << endl;
    cout << (int)index.IsFull() 
         << "|" << index._pPKProfile->getHeader()->usedDocNum 
         << "|" << index._pPKProfile->getHeader()->maxDocNum 
         << "|" << index._pFeatureProfile->getHeader()->usedDocNum 
         << "|" << index._pFeatureProfile->getHeader()->maxDocNum 
         << "|" << index._pIDMap->count()
         << "|" << (int)index._pDeleteMap->test(0) 
         << endl;

    //get 0-10 doc feature
    for(int i = 0; i< 10; i++){
        cout << "i:" << i << " pk:" << index.getPK(i) << endl;
        const char* datas = (const char*)index.getFeature(i);
        for(int j = 0; j < 4; j++){
            float* fea = (float*)(datas + 4 * j);
            cout << setprecision(7) << *fea << ",";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]) {
    // gflags
    gflags::SetUsageMessage("Usage: knn_build_example <plugin files' path>");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
     
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
    
    cout << "index_type:" << FLAGS_index_type << endl;
    ElapsedTime elapsed_time;
    if(FLAGS_index_type == "IVFPQ"){
        IndexIvfpq ivfpq;
        auto file_handle = storage->open(PQ_INDEX_FILENAME, false);
        DoLoad(ivfpq, file_handle);
    }
    else if(FLAGS_index_type == "Flat")
    {
        IndexFlat flat;
        auto file_handle = storage->open(FLAT_INDEX_FILENAME, false);
        DoLoad(flat, file_handle);

    }
    else if(FLAGS_index_type == "PQFlat")
    {
        IndexPqflat pqflat;
        auto file_handle = storage->open(PQFLAT_INDEX_FILENAME, false);
        DoLoad(pqflat, file_handle);
        
        cout << pqflat._pqcodeProfile->getHeader()->usedDocNum 
             << "|" << pqflat._pqcodeProfile->getHeader()->maxDocNum 
             << endl;
    }
    else if(FLAGS_index_type == "IVFFlat")
    {
        IndexIvfflat ivfflat;
        auto file_handle = storage->open(IVFFLAT_INDEX_FILENAME, false);
        DoLoad(ivfflat, file_handle);
        
        cout << ivfflat.get_coarse_index()->getHeader()->slotNum
             << "|" << ivfflat.get_coarse_index()->getHeader()->maxDocSize
             << "|" << ivfflat.get_coarse_index()->getHeader()->capacity
             << "|" << ivfflat.get_coarse_index()->getHeader()->usedSize
             << endl;
    }
    else
    {
        std::cerr << "unknow index type" << endl;
        return -1;
    }
    
    std::cout << "Build & Dump finished, Cost Time:" << elapsed_time.elapsed() << std::endl;
    return 0;
}
