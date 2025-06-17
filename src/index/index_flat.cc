#include "index_flat.h"
#include <iostream>

using namespace std;
using namespace mercury;

bool IndexFlat::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        return false;
    }

    bool res = LoadIndexFromPackage(packageHelper);
    if(!res){
        LOG_ERROR("load flat index failed from hander");
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

bool IndexFlat::Dump(IndexStorage::Pointer storage, const string& file_name, bool only_dump_meta)
{
    IndexPackage packageHelper;

    if(index_params_ == nullptr || !index_params_->has(kDumpDirPathKey)){
        abort();
    }
    string dump_dir_path;
    index_params_->get(kDumpDirPathKey, &dump_dir_path);
    DumpContext dump_context(dump_dir_path);

    bool res = DumpIndexToPackage(packageHelper, only_dump_meta, dump_context);

    if(!res){
        LOG_ERROR("dump flat index failed %s", file_name.c_str());
        return false;
    }

    if (!packageHelper.dump(file_name, storage, false)) {
        LOG_ERROR("flush to storage failed");
        return false;
    }
    return true;
}

bool IndexFlat::Create(IndexStorage::Pointer storage, const string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle)
{
    IndexPackage index_package;
    if (!index_package.load(meta_file_handle, true)) {
        return false;
    }
    
    /// read index meta and L1 quantizer
    bool res = LoadIndexFromPackage(index_package, true);
    // create new segment
    map<string, size_t> stab;
    CreateIndexFromPackage(stab);

    // touch file
    if (!IndexPackage::Touch(file_name, storage, stab)) {
        LOG_ERROR("create segment index package failed!");
        return false;
    }

    // set buffer user new mmap buffer
    stg_handler_ = storage->open(file_name, false);
    if (!stg_handler_) {
        LOG_ERROR("create segment file failed [%s].", file_name.c_str());
        return false;
    }

    // read alloced buffer
    IndexPackage package;
    if (!package.load(stg_handler_, false)) {
        LOG_ERROR("load segment index package failed!");
        return false;
    }
    
    auto *compMeta = package.get(COMPONENT_FEATURE_META);
    if(compMeta == nullptr) {
        LOG_ERROR("get index meta info failed!");
        return false;
    }
    //set data back
    buf_meta_data_.clear();
    index_meta_->serialize(&buf_meta_data_);
    memcpy(compMeta->getData(), buf_meta_data_.data(), buf_meta_data_.size());
    res = index_meta_->deserialize(compMeta->getData(), buf_meta_data_.size());
    if (!res) {
        LOG_ERROR("index meta deserialize error");
        return false;
    }

    auto *compPK = package.get(COMPONENT_PK_PROFILE);
    auto *compFeature = package.get(COMPONENT_FEATURE_PROFILE);
    auto *compIDMap = package.get(COMPONENT_IDMAP);
    auto *compDeleteMap = package.get(COMPONENT_DELETEMAP);
    if (!compPK || !compFeature || !compIDMap || !compDeleteMap) {
        LOG_ERROR("get index component failed!");
        return false;
    }

    res &= _pPKProfile->create(compPK->getData(), compPK->getDataSize(), sizeof(uint64_t));
    res &= _pFeatureProfile->create(compFeature->getData(), compFeature->getDataSize(), feature_info_size_);
    int ret = _pIDMap->mount(reinterpret_cast<char *>(compIDMap->getData()), compIDMap->getDataSize(), doc_num_, true);
    // TODO return value
    _pDeleteMap->mount(compDeleteMap->getData(), compDeleteMap->getDataSize());
    if (!res || ret < 0) {
        LOG_ERROR("create flat index error");
        return false;
    }
    
    return true;
}

int IndexFlat::Add(docid_t doc_id, uint64_t key, const void *val, size_t len)
{
    docid_t tempId = INVALID_DOCID;
    if (_pIDMap->find(key, tempId) && !_pDeleteMap->test(tempId)) {
        LOG_WARN("insert duplicated doc with key[%lu]", key);
        return INVALID_DOCID;
    }

    AddProfile(doc_id, key, val, len);
    return doc_id;
}

bool IndexFlat::RemoveId(uint64_t key)
{
    bool res = false;
    docid_t docId = INVALID_DOCID;
    if (_pIDMap->find(key, docId)) {
        _pDeleteMap->set(docId);
        if (!_pDeleteMap->test(docId)) {
            LOG_ERROR("set pk[%lu] in delete map error.", key);
            return false;
        }
        res = true;
    }
    return res;
}

