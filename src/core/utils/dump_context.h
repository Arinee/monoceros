#ifndef __MERCURY_UTIL_DUMP_CONTEXT_H
#define __MERCURY_UTIL_DUMP_CONTEXT_H

#include <limits>
#include <string>
#include <map>
#include "framework/utility/mmap_file.h"

namespace mercury {

class DumpContext
{
public:
    DumpContext(const std::string& dir_path):dir_path_(dir_path){}
    virtual ~DumpContext(){}

    MMapFile& GetFile(const std::string& file_name){
        return mmap_file_[file_name];
    }

    const std::string& GetDirPath(){
        return dir_path_;
    }

    /// dump profile to package
    inline void DumpPackage(IndexPackage &package, const std::string& componext_name) {
        std::string file_path = dir_path_ + "/" + componext_name;
        MMapFile& mmap_file = GetFile(file_path);
        mmap_file.open(file_path.c_str(), true);
        package.emplace(componext_name, mmap_file.region(), mmap_file.region_size());
    }
private:
    std::map<std::string, MMapFile> mmap_file_;
    std::string dir_path_;
};

}; //namespace mercury

#endif //__MERCURY_UTIL_DUMP_CONTEXT_H
