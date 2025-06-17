#include "cat_index_ivfflat.h"
#include <iostream>

using namespace std;

namespace mercury {

bool CatIndexIvfflat::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        return false;
    }

    bool res = LoadIndexFromPackage(packageHelper);
    res &= LoadIVFIndexFromPackage(packageHelper);
    res &= LoadFLatIndexFromPackage(packageHelper);

    if(!res){
        LOG_ERROR("load ivfflat index failed from hander");
        return false;
    }

    auto* component = packageHelper.get(CAT_COMPONENT_KEY_CAT_MAP);
    if (!component) {
        LOG_ERROR("get component %s error", CAT_COMPONENT_KEY_CAT_MAP.c_str());
        return false;
    }
    if ( 0 != _keyCatMap->mount((char*)component->getData(), component->getDataSize())) {
        LOG_ERROR("failed to load key cat map");
        return false;
    }

    component = packageHelper.get(CAT_COMPONENT_CAT_SET);
    if (!component) {
        LOG_ERROR("get component %s error", CAT_COMPONENT_CAT_SET.c_str());
        return false;
    }
    if ( 0 != _catSet->mount((char*)component->getData(), component->getDataSize())) {
        LOG_ERROR("failed to load cat set");
        return false;
    }

    /// check logic
    if ((size_t)_pFeatureProfile->getHeader()->infoSize 
                != index_meta_->sizeofElement()) {
        LOG_ERROR("check segment failed! feature size[%zd | %zd]",
                _pFeatureProfile->getHeader()->infoSize,
                index_meta_->sizeofElement());
        return false;
    }

    stg_handler_ = move(file_handle);
    return true;
}

bool CatIndexIvfflat::Dump(IndexStorage::Pointer /*storage*/, const string& /*file_name*/, bool /*only_dump_meta*/)
{
    std::cerr << "Not implemented." << std::endl;
    return true;
}

bool CatIndexIvfflat::Create(IndexStorage::Pointer /*storage*/, const string& /*file_name*/, IndexStorage::Handler::Pointer &&/*meta_file_handle*/)
{
    std::cerr << "Not implemented." << std::endl;
    return true;
}

int CatIndexIvfflat::Add(docid_t /*doc_id*/, uint64_t /*key*/, const void * /*val*/, size_t /*len*/)
{
    std::cerr << "Not implemented." << std::endl;
    return -1;
}

bool CatIndexIvfflat::RemoveId(uint64_t /*key*/)
{
    std::cerr << "Not implemented." << std::endl;
    return false;
}

} // namespace mercury
