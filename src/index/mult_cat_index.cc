#include "mult_cat_index.h"
using namespace mercury;
using namespace std;

uint32_t MultCatIndex::GetCatNum(){
    return GetIDMap()->size();
}

void MultCatIndex::MapSlot(int cateId, uint32_t soltId){
    GetIDMap()->insert(cateId, soltId);
}

bool MultCatIndex::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        return false;
    }

    // Read Coarse Index
    auto *component = packageHelper.get(COMPONENT_COARSE_INDEX);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_COARSE_INDEX.c_str());
        return false;
    }
    if (!_coarseIndex->load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("coarse index load error");
        return false;
    }

    // Read IDMap
    component = packageHelper.get(COMPONENT_CATEIDMAP);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_CATEIDMAP.c_str());
        return false;
    }
    if (0 != _IDMap->mount((char*)component->getData(), component->getDataSize())) {
        LOG_ERROR("idmap load error");
        return false;
    }

    stg_handler_ = move(file_handle);
    return true;
}

bool MultCatIndex::Dump(IndexStorage::Pointer storage, const std::string& file_name)
{
    IndexPackage packageHelper;

    if(!index_params_.has(kDumpDirPathKey)){
        abort();
    }
    string dump_dir_path;
    index_params_.get(kDumpDirPathKey, &dump_dir_path);
    DumpContext dump_context(dump_dir_path);

    // write index to disk as segment
    const string& segmentPath = dump_context.GetDirPath();
    bool res = _coarseIndex->dump(segmentPath + "/" + COMPONENT_COARSE_INDEX);
    if (!res) {
        LOG_ERROR("flush data error!");
        return false;
    }
    dump_context.DumpPackage(packageHelper, COMPONENT_COARSE_INDEX);

    int ret = _IDMap->dump(segmentPath + "/" + COMPONENT_CATEIDMAP);
    if (ret != 0) {
        LOG_ERROR("flush data error!");
        return false;
    }
    dump_context.DumpPackage(packageHelper, COMPONENT_CATEIDMAP);
     
    if (!packageHelper.dump(file_name, storage, false)) {
        LOG_ERROR("flush to storage failed");
        return false;
    }
    return true;
}

MultCatIndex::CateFeeder MultCatIndex::GetCateFeeder(int cateId)
{
    return MultCatIndex::CateFeeder(GetMultIndex(), cateId);
}
