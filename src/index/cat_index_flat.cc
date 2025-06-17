#include "cat_index_flat.h"
#include <iostream>

using namespace std;
using namespace mercury;

bool CatIndexFlat::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        LOG_ERROR("failed to load handler");
        return false;
    }

    bool res = LoadIndexFromPackage(packageHelper);
    if(!res){
        LOG_ERROR("load index failed from hander");
        return false;
    }

    if ((size_t)_pFeatureProfile->getHeader()->infoSize
                != index_meta_->sizeofElement()) {
        LOG_ERROR("check segment failed! feature size[%zd | %zd]",
                    _pFeatureProfile->getHeader()->infoSize,
                    index_meta_->sizeofElement());
        return false;
    }

    auto* component = packageHelper.get(CAT_COMPONENT_CAT_SLOT_MAP);
    if (!component) {
        LOG_ERROR("get component %s error", CAT_COMPONENT_CAT_SLOT_MAP.c_str());
        return false;
    }
    if ( 0 != _catSlotMap->mount((char*)component->getData(), component->getDataSize())) {
        LOG_ERROR("failed to load cat slot map");
        return false;
    }

    component = packageHelper.get(CAT_COMPONENT_SLOT_DOC_INDEX);
    if (!component) {
        LOG_ERROR("get component %s error", CAT_COMPONENT_SLOT_DOC_INDEX.c_str());
        return false;
    }
    if (!_slotDocIndex->load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("failed to load slot doc coarse index");
        return false;
    }

    stg_handler_ = move(file_handle);
    return true;
}

bool CatIndexFlat::Dump(IndexStorage::Pointer /*storage*/, const string& /*file_name*/, bool /*only_dump_meta*/)
{
    std::cerr << "Not implemented..." << std::endl;
    return false;
}

bool CatIndexFlat::Create(IndexStorage::Pointer /*storage*/, const string& /*file_name*/, IndexStorage::Handler::Pointer &&/*meta_file_handle*/)
{
    std::cerr << "Not implemented...." << std::endl;
    return false;
}

int CatIndexFlat::Add(docid_t /*doc_id*/, uint64_t /*key*/, const void * /*val*/, size_t /*len*/)
{
    std::cerr << "Not implemented...." << std::endl;
    return -1;
}

bool CatIndexFlat::RemoveId(uint64_t /*key*/)
{
    std::cerr << "Not implemented...." << std::endl;
    return false;
}

