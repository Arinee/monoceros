#include "cat_index_ivfpq.h"

using namespace std;
using namespace mercury;

bool CatIndexIvfpq::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        return false;
    }

    bool res = LoadIndexFromPackage(packageHelper);
    res &= LoadIVFIndexFromPackage(packageHelper);
    res &= LoadFLatIndexFromPackage(packageHelper);
    res &= LoadPQIndexFromPackage(packageHelper);

    if(!res){
        LOG_ERROR("load ivfpq index failed ");
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

    if ((size_t)_pFeatureProfile->getHeader()->infoSize 
                != index_meta_->sizeofElement()
            || (size_t)_pqcodeProfile->getHeader()->infoSize 
                != getCentroidResource()->getIntegrateMeta().fragmentNum * sizeof(uint16_t)) {
        LOG_ERROR("check segment failed! feature size[%zd | %zd], product size[%zd | %zd]",
                _pFeatureProfile->getHeader()->infoSize,
                index_meta_->sizeofElement(),
                _pqcodeProfile->getHeader()->infoSize,
                  getCentroidResource()->getIntegrateMeta().fragmentNum * sizeof(uint16_t));
        return false;
    }

    stg_handler_ = move(file_handle);
    return true;
}
