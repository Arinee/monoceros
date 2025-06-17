#include "index_ivfflat.h"
#include <iostream>

using namespace std;

namespace mercury {

bool IndexIvfflat::Load(IndexStorage::Handler::Pointer &&file_handle)
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

bool IndexIvfflat::Dump(IndexStorage::Pointer storage, const string& file_name, bool only_dump_meta)
{
    IndexPackage packageHelper;

    if(index_params_ == nullptr || !index_params_->has(kDumpDirPathKey)){
        abort();
    }
    string dump_dir_path;
    index_params_->get(kDumpDirPathKey, &dump_dir_path);
    DumpContext dump_context(dump_dir_path);

    bool res = DumpIndexToPackage(packageHelper, only_dump_meta, dump_context);
    res &= DumpIVFIndexToPackage(packageHelper, only_dump_meta);
    res &= DumpFLatIndexToPackage(packageHelper, only_dump_meta, dump_context);

    if(!res){
        LOG_ERROR("dump ivfflat index failed %s", file_name.c_str());
        return false;
    }

    if (!packageHelper.dump(file_name, storage, false)) {
        LOG_ERROR("flush to storage failed");
        return false;
    }
    return true;
}

bool IndexIvfflat::Create(IndexStorage::Pointer storage, const string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle)
{
    IndexPackage index_package;
    if (!index_package.load(meta_file_handle, false)) {
        return false;
    }
    
    /// read index meta and L1 quantizer
    bool res = LoadIndexFromPackage(index_package, true);
    res &= _centroidQuantizer->LoadLevelOneQuantizer(index_package);
    if(!res){
        LOG_ERROR("read meta index failed!");
        return false;
    }

    // read slotNum first
    slot_num_ = _centroidQuantizer->get_slot_num();
    // create new segment
    map<string, size_t> stab;
    CreateIndexFromPackage(stab);
    CreateIVFIndexFromPackage(stab);
    CreateFLatIndexFromPackage(stab);

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
    auto *compCentroidRough = package.get(COMPONENT_ROUGH_MATRIX);
    auto *compCentroidIntegrate = package.get(COMPONENT_INTEGRATE_MATRIX);
    if(compMeta == nullptr || compCentroidRough == nullptr || compCentroidIntegrate == nullptr){
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

    string roughMatrix;
    auto* centroid_quantizer = get_centroid_quantizer();
    centroid_quantizer->get_centroid_resource()->dumpRoughMatrix(roughMatrix);
    memcpy(compCentroidRough->getData(), roughMatrix.data(), roughMatrix.size());
    res = centroid_quantizer->get_centroid_resource()->init((void *)compCentroidRough->getData(),
            compCentroidRough->getDataSize(),
            (void *)compCentroidIntegrate->getData(), 
            compCentroidIntegrate->getDataSize());
    if (!res) {
        LOG_ERROR("centroid resource init error");
        return false;
    }

    auto *compIndex = package.get(COMPONENT_COARSE_INDEX);
    auto *compPK = package.get(COMPONENT_PK_PROFILE);
    auto *compFeature = package.get(COMPONENT_FEATURE_PROFILE);
    auto *compIDMap = package.get(COMPONENT_IDMAP);
    auto *compDeleteMap = package.get(COMPONENT_DELETEMAP);
    if (!compIndex || !compPK || !compFeature || !compIDMap || !compDeleteMap) {
        LOG_ERROR("get index component failed!");
        return false;
    }

    res = coarse_index_->create(compIndex->getData(), compIndex->getDataSize(), slot_num_, doc_num_);
    res &= _pPKProfile->create(compPK->getData(), compPK->getDataSize(), sizeof(uint64_t));
    res &= _pFeatureProfile->create(compFeature->getData(), compFeature->getDataSize(), feature_info_size_);
    int ret = _pIDMap->mount(reinterpret_cast<char *>(compIDMap->getData()), compIDMap->getDataSize(), doc_num_, true);
    _pDeleteMap->mount(compDeleteMap->getData(), compDeleteMap->getDataSize());
    if (!res || ret < 0) {
        LOG_ERROR("create ivfflat index error");
        return false;
    }
    
    return true;
}

int IndexIvfflat::Add(docid_t doc_id, uint64_t key, const void *val, size_t len)
{
    docid_t tempId = INVALID_DOCID;
    if (_pIDMap->find(key, tempId) && !_pDeleteMap->test(tempId)) {
        LOG_WARN("insert duplicated doc with key[%lu]", key);
        return INVALID_DOCID;
    }

    // calc label
    vector<size_t> levelScanLimit;
    for (size_t i = 0; i < getCentroidResource()->getRoughMeta().levelCnt - 1; ++i) {
        levelScanLimit.push_back(getCentroidResource()->getRoughMeta().centroidNums[i] / 10);
    }
    int32_t label = _centroidQuantizer->CalcLabel(val, len, index_meta_, levelScanLimit);

    bool res = coarse_index_->addDoc(label, doc_id);
    if (!res) {
        LOG_ERROR("add vector to segment error with key[%lu] docid[%d]", key, doc_id);
        return INVALID_DOCID;
    }

    AddProfile(doc_id, key, val, len);
    return doc_id;
}

bool IndexIvfflat::RemoveId(uint64_t key)
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

bool IndexIvfflat::CreateFLatIndexFromPackage(map<string, size_t>& stab)
{
    size_t indexCapacity = CoarseIndex::calcSize(slot_num_, doc_num_);

    if (indexCapacity <= 0) {
        LOG_ERROR("calculate index size error!");
        return false;
    }

    stab.emplace(COMPONENT_COARSE_INDEX, indexCapacity);
    return true;
} 

bool IndexIvfflat::LoadFLatIndexFromPackage(IndexPackage &package)
{
    // Read Coarse Index
    auto *component = package.get(COMPONENT_COARSE_INDEX);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_COARSE_INDEX.c_str());
        return false;
    }
    if (!coarse_index_->load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("coarse index load error");
        return false;
    }

    return true;
}

bool IndexIvfflat::DumpFLatIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& dump_context)
{
    // dump data
    if(!only_dump_meta)
    {
        // size_t maxDocNum = _pPKProfile->getHeader()->usedDocNum;
        // package.emplace(COMPONENT_COARSE_INDEX, coarse_index_->getHeader(), indexCapacity);
        // write index to disk as segment
        const string& segmentPath = dump_context.GetDirPath();
        bool res = coarse_index_->dump(segmentPath + "/" + COMPONENT_COARSE_INDEX);
        if (!res) {
            LOG_ERROR("flush data error!");
            return false;
        }

        dump_context.DumpPackage(package, COMPONENT_COARSE_INDEX);
        // MMapFile& index_file = dump_context.GetFile(segmentPath + "/" + COMPONENT_COARSE_INDEX);
        // index_file.open(string(segmentPath + "/" + COMPONENT_COARSE_INDEX).c_str(), true);
        // package.emplace(COMPONENT_COARSE_INDEX, index_file.region(), index_file.region_size());
    }

    return true;
}

} // namespace mercury
