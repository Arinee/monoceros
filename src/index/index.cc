#include "index.h"
#include <iostream>
using namespace std;
using namespace mercury;

bool Index::LoadIndexFromPackage(IndexPackage &package, bool only_dump_meta)
{
    // read index meta
    if (!index_meta_) 
    {
        auto *component = package.get(COMPONENT_FEATURE_META);
        if (!component) {
            LOG_ERROR("get component %s error", COMPONENT_FEATURE_META.c_str());
            return false;
        }
        index_meta_ = new mercury::IndexMeta();
        if (!index_meta_->deserialize(component->getData(), component->getDataSize())) {
            LOG_ERROR("index meta deserialize error");
            return false;
        }
        /*
           if (_searchMethod != IndexDistance::Methods::kMethodUnknown) {
           LOG_INFO("use online search method instand offline index meta");
           _pIndexMeta->setMethod(_searchMethod);
           }*/
    }

    if(only_dump_meta){
        return true;
    }

    // Read pk profile
    auto *component = package.get(COMPONENT_PK_PROFILE);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_PK_PROFILE.c_str());
        return false;
    }
    if (!_pPKProfile->load((void*)component->getData(), component->getDataSize())) {
        LOG_ERROR("pk profile load error");
        return false;
    }

    // Read feature profile
    component = package.get(COMPONENT_FEATURE_PROFILE);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_FEATURE_PROFILE.c_str());
        return false;
    }
    if (!_pFeatureProfile->load(component->getData(), component->getDataSize())) {
        LOG_ERROR("feature profile load error");
        return false;
    }

    // Read IDMap
    component = package.get(COMPONENT_IDMAP);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_IDMAP.c_str());
        return false;
    }
    if (0 != _pIDMap->mount((char*)component->getData(), component->getDataSize())) {
        LOG_ERROR("idmap load error");
        return false;
    }

    // Read DeleteMap
    component = package.get(COMPONENT_DELETEMAP);
    if (component) {
        _pDeleteMap->mount((void*)component->getData(), component->getDataSize()); // return void
    } else {
        LOG_WARN("get component %s error, create deletemap in memory.", COMPONENT_DELETEMAP.c_str());
        size_t delMapSize = BitsetHelper::CalcBufferSize(_pPKProfile->getHeader()->maxDocNum);
        shared_ptr<char> data(new char[delMapSize], std::default_delete<char[]>());
        _pDeleteMap->mount((void*)data.get(), delMapSize);
    }

    if (index_meta_ == nullptr) {
        LOG_ERROR("meta nullptr error!");
        return false;
    }

    // check profile num
    if ((size_t)_pFeatureProfile->getHeader()->infoSize 
            != index_meta_->sizeofElement()) {
        LOG_ERROR("check segment failed! feature size[%zd | %zd]",
                _pFeatureProfile->getHeader()->infoSize,
                index_meta_->sizeofElement());
        return false;
    }

    feature_info_size_ = _pFeatureProfile->getHeader()->infoSize;
    doc_num_ = _pFeatureProfile->getHeader()->usedDocNum;

    return true;
}

bool Index::DumpIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& dump_context)
{
    if (index_meta_ == nullptr) {
        LOG_ERROR("meta nullptr error!");
        return false;
    }

    buf_meta_data_.clear();
    index_meta_->serialize(&buf_meta_data_);
    package.emplace(COMPONENT_FEATURE_META, buf_meta_data_.data(), buf_meta_data_.size());

    // dump data
    if(!only_dump_meta)
    {
        //size_t maxDocNum = _pPKProfile->getHeader()->usedDocNum;
        //int64_t pkCapacity = ArrayProfile::CalcSize(maxDocNum, _pPKProfile->getHeader()->infoSize);
        //int64_t featureCapacity = ArrayProfile::CalcSize(maxDocNum, _pFeatureProfile->getHeader()->infoSize);
        //uint64_t idMapCapacity = HashTable<uint64_t, docid_t>::needMemSize(maxDocNum);
        size_t delMapSize = BitsetHelper::CalcBufferSize(_pPKProfile->getHeader()->maxDocNum);

        // write index to disk as segment
        const string& segmentPath = dump_context.GetDirPath();
        bool res = _pPKProfile->dump(segmentPath + "/" + COMPONENT_PK_PROFILE);
        res &= _pFeatureProfile->dump(segmentPath + "/" + COMPONENT_FEATURE_PROFILE);
        int ret = _pIDMap->dump(segmentPath + "/" + COMPONENT_IDMAP);
        if (!res || ret != 0) {
            LOG_ERROR("flush data error!");
            return false;
        }

        dump_context.DumpPackage(package, COMPONENT_PK_PROFILE);
        dump_context.DumpPackage(package, COMPONENT_FEATURE_PROFILE);
        dump_context.DumpPackage(package, COMPONENT_IDMAP);
        package.emplace(COMPONENT_DELETEMAP, _pDeleteMap->getBase(), delMapSize);

        // MMapFile& pk_file = dump_context.GetFile(segmentPath + "/" + COMPONENT_PK_PROFILE);
        // pk_file.open(string(segmentPath + "/" + COMPONENT_PK_PROFILE).c_str(), true);
        // package.emplace(COMPONENT_PK_PROFILE, pk_file.region(), pk_file.region_size());

        // MMapFile& pf_file = dump_context.GetFile(segmentPath + "/" + COMPONENT_FEATURE_PROFILE);
        // pf_file.open(string(segmentPath + "/" + COMPONENT_FEATURE_PROFILE).c_str(), true);
        // package.emplace(COMPONENT_FEATURE_PROFILE, pf_file.region(), pf_file.region_size());

        // MMapFile& id_file = dump_context.GetFile(segmentPath + "/" + COMPONENT_IDMAP);
        // id_file.open(string(segmentPath + "/" + COMPONENT_IDMAP).c_str(), true);
        // package.emplace(COMPONENT_IDMAP, id_file.region(), id_file.region_size());

        // cout << pk_file.region_size() << "|" 
        //     << pf_file.region_size() << "|"
        //     << id_file.region_size() << "|"
        //     << delMapSize << endl;
    }

    return true;
}

bool Index::CreateIndexFromPackage(std::map<std::string, size_t>& stab)
{
    if (index_meta_ == nullptr) {
        LOG_ERROR("meta nullptr error!");
        return false;
    }

    ReaIndexParams();
    buf_meta_data_.clear();
    index_meta_->serialize(&buf_meta_data_);

    stab.emplace(COMPONENT_FEATURE_META, buf_meta_data_.size());

    int64_t pkCapacity = ArrayProfile::CalcSize(doc_num_, sizeof(uint64_t));
    int64_t featureCapacity = ArrayProfile::CalcSize(doc_num_, feature_info_size_);
    uint64_t idMapCapacity = HashTable<uint64_t, docid_t>::needMemSize(doc_num_);
    size_t delMapCapacity = BitsetHelper::CalcBufferSize(doc_num_);

    // cout << pkCapacity << "|" 
    //     << featureCapacity << "|"
    //     << idMapCapacity << "|"
    //     << delMapCapacity << endl;

    if (pkCapacity <= 0 || featureCapacity <= 0 
            || idMapCapacity <= 0 || delMapCapacity <= 0) {
        LOG_ERROR("calculate component size error!");
        return false;
    }

    stab.emplace(COMPONENT_PK_PROFILE, pkCapacity);
    stab.emplace(COMPONENT_FEATURE_PROFILE, featureCapacity);
    stab.emplace(COMPONENT_IDMAP, idMapCapacity);
    stab.emplace(COMPONENT_DELETEMAP, delMapCapacity);
    return true;
}

//TODO mem leak
void Index::set_index_meta(IndexMeta* index_meta)
{   
    index_meta_ = index_meta;
}

const IndexMeta* Index::get_index_meta() const
{
    return index_meta_;
}

// TODO new outside
void Index::set_index_params(IndexParams* index_params)
{   
    index_params_ = index_params;
}

IndexParams* Index::get_index_params()
{
    return index_params_;
}

uint64_t Index::getPK(docid_t Doc_Id)
{
    const uint64_t *pk = reinterpret_cast<const uint64_t *>(_pPKProfile->getInfo(Doc_Id));
    return *pk;
}

const void* Index::getFeature(docid_t Doc_Id)
{
    const void *feature = _pFeatureProfile->getInfo(Doc_Id);
    return feature;
}

void Index::ReaIndexParams()
{
    if(index_params_){
        index_params_->get(kBuildDocNumKey, &doc_num_);
        index_params_->get(kFeatureInfoSizeKey, &feature_info_size_);
    }
}

bool Index::IsFull()
{
    if( unlikely( _pFeatureProfile->getHeader()->usedDocNum >= _pFeatureProfile->getHeader()->maxDocNum) ){
        return true;
    }

    return false;
}

int Index::AddProfile(docid_t doc_id, uint64_t key, const void *val, size_t /*len*/)
{
    bool res = _pPKProfile->insert(doc_id, &key);
    res &= _pFeatureProfile->insert(doc_id, val);
    int ret = _pIDMap->insert(key, doc_id);
    //reset deletemap
    _pDeleteMap->reset(doc_id);
    if (!res || ret != 0) {
        LOG_ERROR("add vector to segment error with key[%lu] docid[%d]", key, doc_id);
        return PROFILE_ADD_ERROR;
    }

    return 0;
}