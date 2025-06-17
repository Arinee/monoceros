#include "base_index_provider.h"
#include "framework/utility/file.h"
#include <iostream>

using namespace std;

namespace mercury {

bool BaseIndexProvider::init(size_t incrSegmentDocNum, 
        const std::string &incrSegmentPath,
        std::shared_ptr<IndexParams> index_params_pointer)
{
    // TODO _incrSegmentDocNum ??
    _incrSegmentDocNum = incrSegmentDocNum;
    _incrSegmentPath = incrSegmentPath;

    if(index_params_pointer == nullptr){
        LOG_ERROR("index params must set!");
        abort();
        return false;
    }
    index_params_pointer_ = index_params_pointer;

    return true;
}

bool BaseIndexProvider::load(IndexStorage::Handler::Pointer &&handlerPtr, shared_ptr<Index> segmentPtr)
{
    // Read package as segment
    if (!segmentPtr->Load(move(handlerPtr))) {
        LOG_ERROR("load segment failed!");
        return false;
    }

    _segments.emplace_back(segmentPtr);
    _lastSegment = segmentPtr;
    if (segmentPtr->_pFeatureProfile->getHeader()) {
        _incrId = segmentPtr->_pFeatureProfile->getHeader()->usedDocNum;
    }
    _lastSegment->set_index_params(new IndexParams(*(index_params_pointer_.get())));
    return true;
}

bool BaseIndexProvider::unload()
{
    // unload all segments
    _segments.clear();
    _lastSegment.reset();
    _incrFileHolder.clear();
    return true;
}

key_t BaseIndexProvider::getPK(gloid_t gloid) const
{
    segid_t segid = GET_SEGID(gloid);
    docid_t docid = GET_DOCID(gloid);
    if (likely(segid < _segments.size())) {
        return _segments[segid]->getPK(docid);
    } else {
        return INVALID_KEY;
    }
}

const void *BaseIndexProvider::getFeature(gloid_t gloid) const
{
    segid_t segid = GET_SEGID(gloid);
    docid_t docid = GET_DOCID(gloid);
    if (likely(segid < _segments.size())) {
        return _segments[segid]->getFeature(docid);
    } else {
        return nullptr;
    }
}

gloid_t BaseIndexProvider::addVector(key_t key, const void *feature, size_t len)
{
    // TODO empty is not ok?
    if (_segments.empty()) {
        return INVALID_GLOID;;
    }
    if (feature == nullptr) {
        LOG_ERROR("Add vector can't be nullptr");
        return INVALID_GLOID;
    }

    for (size_t i = 0; i < _segments.size(); ++i) {
        docid_t docId = INVALID_DOCID;
        if (_segments[i]->_pIDMap->find(key, docId) 
                && !_segments[i]->_pDeleteMap->test(docId)) {
            LOG_ERROR("add duplicated doc with key[%lu]", key);
            return INVALID_GLOID;
        }
    }

    lock_guard<mutex> lock(_addLock);
    if (_lastSegment == nullptr || _lastSegment->IsFull()) {
        shared_ptr<Index> alloc_index(_lastSegment->CloneEmptyIndex());
        if (!createSegment(alloc_index)) {
            LOG_ERROR("create new segment failed!");
            return INVALID_GLOID;
        }
        _segments.emplace_back(_lastSegment);
        _incrId = 0;
        LOG_DEBUG("create new segment done!");
    }
    
    segid_t segid = _segments.size() - 1;
    docid_t docid = _lastSegment->Add(_incrId, key, feature, len);
    if (docid == INVALID_DOCID) {
        LOG_ERROR("Add vector to segment failed!");
        return INVALID_GLOID;
    }
    _incrId = docid + 1;
    gloid_t gloid = GET_GLOID(segid, docid); 
    LOG_DEBUG("Add vector success with key[%lu] segid[%u] docid[%u] gloid[%lu]", 
            key, segid, docid, gloid);

    return gloid;
}

bool BaseIndexProvider::deleteVector(key_t key)
{
    if (_segments.empty()) {
        return false;
    }
    bool res = false;
    //TODO why iterate all segment
    for (size_t i = 0; i < _segments.size(); ++i) {
        docid_t docId = INVALID_DOCID;
        if (_segments[i]->_pIDMap->find(key, docId)) {
            _segments[i]->_pDeleteMap->set(docId);
            if (!_segments[i]->_pDeleteMap->test(docId)) {
                LOG_ERROR("set pk[%lu] in delete map error.", key);
                return false;
            }
            res = true;
        }
    }
    return res;
}

bool BaseIndexProvider::createSegment(shared_ptr<Index> segmentPtr)
{
    if (_incrSegmentPath.empty() || _incrSegmentDocNum <= 0 
        || _lastSegment == nullptr || index_params_pointer_ == nullptr) {
        LOG_ERROR("not init");
        return false;
    }
    if (!File::MakePath(_incrSegmentPath.c_str())) {
        LOG_ERROR("make incr segment directory[%s] failed!", _incrSegmentPath.c_str());
        return false;
    }
    string segmentPath = _incrSegmentPath + "/segment_" + to_string(time(nullptr));

    IndexStorage::Pointer stg = InstanceFactory::CreateStorage("MMapFileStorage");
    if (!stg) {
        LOG_ERROR("create storage failed.");
        return false;
    }

    //prepare data
    string tmpMetaPath = _incrSegmentPath + "/tmp_meta";
    bool res = _lastSegment->Dump(stg, tmpMetaPath, true);
    if(!res){
        LOG_ERROR("dump meta failed.");
        return false;
    }

    IndexStorage::Handler::Pointer meta_file_handle = stg->open(tmpMetaPath, false);
    if (!meta_file_handle) {
        LOG_ERROR("create segment file failed [%s].", tmpMetaPath.c_str());
        return false;
    }
    
    //set param
    segmentPtr->set_index_params(new IndexParams(*_lastSegment->get_index_params()));
    res = segmentPtr->Create(stg, segmentPath, move(meta_file_handle));

    if(!res){
        LOG_ERROR("create IndexIvfFlat failed.");
        return false;
    }
    _lastSegment = segmentPtr;
    return true;
}

} // namespace mercury
