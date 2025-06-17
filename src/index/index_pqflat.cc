#include "index_pqflat.h"

using namespace std;
namespace mercury {

//implement index pqflat
void IndexPqflat::ReadPQParams()
{
    auto* centroid_resource = get_centroid_quantizer()->get_centroid_resource();
    uint32_t fragmentNum = centroid_resource->getIntegrateMeta().fragmentNum;
    product_info_size_ = sizeof(uint16_t) * fragmentNum;

    if(product_info_size_ <= 0){
        abort();
    }
}

bool IndexPqflat::Load(IndexStorage::Handler::Pointer &&file_handle)
{
    IndexPackage packageHelper;
    if (!packageHelper.load(file_handle, false)) {
        return false;
    }

    bool res = LoadIndexFromPackage(packageHelper);
    res &= LoadIVFIndexFromPackage(packageHelper);
    res &= LoadPQIndexFromPackage(packageHelper);

    if(!res){
        LOG_ERROR("load index(pqflat) failed ");
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

bool IndexPqflat::Dump(IndexStorage::Pointer storage, const string& file_name, bool only_dump_meta)
{
    IndexPackage packageHelper;
    string dump_dir_path;
    index_params_->get(kDumpDirPathKey, &dump_dir_path);
    DumpContext dump_context(dump_dir_path);

    bool res = DumpIndexToPackage(packageHelper, only_dump_meta, dump_context);
    res &= DumpIVFIndexToPackage(packageHelper, only_dump_meta);
    res &= DumpPQIndexToPackage(packageHelper, only_dump_meta, dump_context);

    if(!res){
        LOG_ERROR("dump pqflat index failed %s", file_name.c_str());
        return false;
    }

    if (!packageHelper.dump(file_name, storage, false)) {
        LOG_ERROR("flush to storage failed");
        return false;
    }
    return true;
}

bool IndexPqflat::Create(IndexStorage::Pointer storage, const string& file_name, IndexStorage::Handler::Pointer &&meta_file_handle)
{
    IndexPackage meta_package;
    if (!meta_package.load(meta_file_handle, false)) {
        return false;
    }

    /// read index meta and L1 quantizer
    bool res = _centroidQuantizer->LoadLevelOneQuantizer(meta_package);
    res &= LoadIndexFromPackage(meta_package, true);
    if(!res){
        LOG_ERROR("read meta index failed!");
        return false;
    }

    // create new segment
    map<string, size_t> stab;
    CreateIndexFromPackage(stab);
    CreateIVFIndexFromPackage(stab);
    CreatePQIndexFromPackage(stab);

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
    auto *compPK = package.get(COMPONENT_PK_PROFILE);
    auto *compProduct = package.get(COMPONENT_PRODUCT_PROFILE);
    auto *compFeature = package.get(COMPONENT_FEATURE_PROFILE);
    auto *compIDMap = package.get(COMPONENT_IDMAP);
    auto *compDeleteMap = package.get(COMPONENT_DELETEMAP);
    if (!compPK || !compFeature || !compIDMap 
            || !compDeleteMap || !compProduct 
            || !compCentroidRough || !compCentroidIntegrate || !compMeta) {
        LOG_ERROR("get component failed!");
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

    string roughMatrix, integrateMatrix;
    auto* centroid_quantizer = get_centroid_quantizer();
    centroid_quantizer->get_centroid_resource()->dumpRoughMatrix(roughMatrix);
    memcpy(compCentroidRough->getData(), roughMatrix.data(), roughMatrix.size());

    centroid_quantizer->get_centroid_resource()->dumpIntegrateMatrix(integrateMatrix);
    memcpy(compCentroidIntegrate->getData(), integrateMatrix.data(), integrateMatrix.size());   
    
    res = centroid_quantizer->get_centroid_resource()->init((void *)compCentroidRough->getData(),
            compCentroidRough->getDataSize(),
            (void *)compCentroidIntegrate->getData(), 
            compCentroidIntegrate->getDataSize());
    if (!res) {
        LOG_ERROR("centroid resource init error");
        return false;
    }

    res &= _pPKProfile->create(compPK->getData(), compPK->getDataSize(), sizeof(uint64_t));
    res &= _pqcodeProfile->create(compProduct->getData(), compProduct->getDataSize(), product_info_size_);
    res &= _pFeatureProfile->create(compFeature->getData(), compFeature->getDataSize(), feature_info_size_);
    int ret = _pIDMap->mount(reinterpret_cast<char *>(compIDMap->getData()), compIDMap->getDataSize(), doc_num_, true);
    _pDeleteMap->mount(compDeleteMap->getData(), compDeleteMap->getDataSize());
    if (!res || ret < 0) {
        LOG_ERROR("create pqflat index error");
        return false;
    }

    return true;
}

bool IndexPqflat::CreatePQIndexFromPackage(map<string, size_t>& stab)
{
    ReadPQParams();
    int64_t productCapacity = ArrayProfile::CalcSize(doc_num_, product_info_size_);

    if (productCapacity <= 0) {
        LOG_ERROR("calculate product component size error!");
        return false;
    }

    stab.emplace(COMPONENT_PRODUCT_PROFILE, productCapacity);
    return true;
} 

bool IndexPqflat::LoadPQIndexFromPackage(IndexPackage &package)
{
    // Read feature profile
    auto *component = package.get(COMPONENT_PRODUCT_PROFILE);
    if (!component) {
        LOG_ERROR("get component %s error", COMPONENT_PRODUCT_PROFILE.c_str());
        return false;
    }
    if (!_pqcodeProfile->load(component->getData(), component->getDataSize())) {
        LOG_ERROR("feature profile load error");
        return false;
    }

    product_info_size_ = _pqcodeProfile->getHeader()->infoSize;
    return true;
}

bool IndexPqflat::DumpPQIndexToPackage(IndexPackage &package, bool only_dump_meta, DumpContext& /*dump_context*/)
{
    // dump data
    if(!only_dump_meta)
    {
        size_t maxDocNum = _pPKProfile->getHeader()->maxDocNum;
        int64_t productCapacity = ArrayProfile::CalcSize(maxDocNum, product_info_size_);
        
        package.emplace(COMPONENT_PRODUCT_PROFILE, _pPKProfile->getHeader(), productCapacity);
    }

    return true;
}

vector<size_t> IndexPqflat::computeRoughLevelScanLimit(GeneralSearchContext* context)
{
    float roughLevelScanRatio = context->getLevelScanRatio();
    if (context && context->updateLevelScanParam()) {
        roughLevelScanRatio = context->getLevelScanRatio();
    }

    size_t levelCentroidNum = getCentroidResource()->getRoughMeta().centroidNums[0]; // level_0
    float levelScanLimit = roughLevelScanRatio * levelCentroidNum;
    return {(size_t)(levelScanLimit < 1 ? levelCentroidNum : levelScanLimit)};
}

QueryDistanceMatrix::Pointer 
IndexPqflat::InitQueryDistanceMatrix(const void *query, GeneralSearchContext *context)
{
    CentroidResource* centroid_resource = getCentroidResource();
    if(centroid_resource == nullptr){
        LOG_ERROR("CentroidResource nullptr error");
        return QueryDistanceMatrix::Pointer();
    }

    QueryDistanceMatrix::Pointer qdm;
    qdm.reset(new QueryDistanceMatrix(*index_meta_, centroid_resource));
    vector<size_t> roughLevelScanLimit = computeRoughLevelScanLimit(context);
    if (!qdm->init(query, roughLevelScanLimit)) {
        LOG_ERROR("create query distance matrix error");
        return QueryDistanceMatrix::Pointer();
    }
    return qdm;
}

int IndexPqflat::Add(docid_t doc_id, uint64_t key, const void *val, size_t len)
{
    // calculate the code feature
    CentroidResource* centroid_resource = getCentroidResource();
    QueryDistanceMatrix qdm(*index_meta_, centroid_resource);
    // use 10% as level scan limit
    vector<size_t> levelScanLimit;
    for (size_t i = 0; i < centroid_resource->getRoughMeta().levelCnt - 1; ++i) {
        levelScanLimit.push_back(centroid_resource->getRoughMeta().centroidNums[i] / 10);
    }
    bool res = qdm.init(val, levelScanLimit, true);
    if (!res) {
        LOG_ERROR("Calcualte QDM failed!");
        return PROFILE_ADD_ERROR;
    }

    vector<uint16_t> productLabels;
    if (!qdm.getQueryCodeFeature(productLabels)) {
        LOG_ERROR("get query codefeature failed!");
        return PROFILE_ADD_ERROR;
    }
    
    res = _pqcodeProfile->insert(doc_id, &productLabels[0]);
    if (!res) {
        LOG_ERROR("add product error with key[%lu] docid[%d]", key, doc_id);
        return PROFILE_ADD_ERROR;
    }

    AddProfile(doc_id, key, val, len);
    return doc_id;
}

bool IndexPqflat::RemoveId(uint64_t key)
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


} // namespace mercury
