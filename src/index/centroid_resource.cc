#include "centroid_resource.h"
#include <string.h>
#include "framework/index_logger.h"
#include "utils/string_util.h"

using namespace std;

namespace mercury {

CentroidResource::~CentroidResource()
{
    if (_memAlloc) {
        if (_roughMatrix != nullptr) {
            delete[] _roughMatrix;
            _roughMatrix = nullptr;
        }
        if (_integrateMatrix != nullptr) {
            delete[] _integrateMatrix;
            _integrateMatrix = nullptr;
        }
    }
}

bool CentroidResource::init(void *pRoughBase, size_t roughLen)
{
    _roughOnly = true;
    _memAlloc = false;
    if (!parseRoughContent((char *)pRoughBase, roughLen)) {
        return false;
    }
    return validate();
}

bool CentroidResource::init(void *pRoughBase, size_t roughLen,
        void *pIntegrateBase, size_t integrateLen)
{
    _memAlloc = false;
    if (!parseRoughContent((char *)pRoughBase, roughLen)) {
        return false;
    }
    if (!parseIntegrateContent((char* )pIntegrateBase, integrateLen)) {
        //TODO: spilt rough and integrate
        return false;
    }
    return validate();
}

bool CentroidResource::create(const RoughMeta& roughMeta)
{
    _roughOnly = true;
    // check first
    if (roughMeta.levelCnt != roughMeta.centroidNums.size()) {
        return false;
    }
    _roughMeta = roughMeta;
    // clean old data
    if (_roughMatrix != nullptr) {
        delete[] _roughMatrix;
        _roughMatrix = nullptr;
    }

    _memAlloc = true;
    size_t centroidNumCnt = 0;
    size_t ratio = 1;
    for (size_t l = 0; l < _roughMeta.levelCnt; l++) {
        centroidNumCnt += _roughMeta.centroidNums[l] * ratio;
        ratio *= _roughMeta.centroidNums[l];
    }
    _roughMatrixSize = centroidNumCnt * roughMeta.elemSize;
    _roughMatrix = new (std::nothrow) char[_roughMatrixSize];
    if (!_roughMatrix) {
        return false;
    }
    memset(_roughMatrix, 0, _roughMatrixSize);

    return true;
}

bool CentroidResource::create(const RoughMeta& roughMeta, 
        const IntegrateMeta& integrateMeta)
{
    // check first
    if (roughMeta.levelCnt != roughMeta.centroidNums.size()) {
        return false;
    }
    _roughMeta = roughMeta;
    _integrateMeta = integrateMeta;
    // clean old data
    if (_roughMatrix != nullptr) {
        delete[] _roughMatrix;
        _roughMatrix = nullptr;
    }
    if (_integrateMatrix != nullptr) {
        delete[] _integrateMatrix;
        _integrateMatrix = nullptr;
    }

    _memAlloc = true;
    size_t centroidNumCnt = 0;
    size_t ratio = 1;
    for (size_t l = 0; l < _roughMeta.levelCnt; l++) {
        centroidNumCnt += _roughMeta.centroidNums[l] * ratio;
        ratio *= _roughMeta.centroidNums[l];
    }
    _roughMatrixSize = centroidNumCnt * roughMeta.elemSize;
    _roughMatrix = new (std::nothrow) char[_roughMatrixSize];
    if (!_roughMatrix) {
        return false;
    }
    memset(_roughMatrix, 0, _roughMatrixSize);

    _integrateMatrixSize = integrateMeta.fragmentNum * integrateMeta.centroidNum * integrateMeta.elemSize;
    _integrateMatrix = new (std::nothrow) char[_integrateMatrixSize];
    if (!_integrateMatrix) {
        return false;
    }
    memset(_integrateMatrix, 0, _integrateMatrixSize);

    return true;
}

void CentroidResource::dumpRoughMatrix(string& roughString) const
{
    // dump rough matrix
    roughString.reserve(3 * sizeof(uint32_t) + _roughMeta.levelCnt * sizeof(uint32_t) + _roughMatrixSize);
    roughString = string((char*)&(_roughMeta.magic), sizeof(uint32_t));
    roughString.append((char*)&(_roughMeta.elemSize), sizeof(uint32_t));
    roughString.append((char*)&(_roughMeta.levelCnt), sizeof(uint32_t));
    for (auto centroidNum : _roughMeta.centroidNums) {
        roughString.append((char*)&centroidNum, sizeof(uint32_t));
    }
    roughString.append((char*)_roughMatrix, _roughMatrixSize);
}

void CentroidResource::dumpIntegrateMatrix(string& integrateString) const
{
    // dump integrate matrix
    integrateString.reserve(4 * sizeof(uint32_t) + _integrateMatrixSize);
    integrateString = string((char*)&(_integrateMeta.magic), sizeof(uint32_t));
    integrateString.append((char*)&(_integrateMeta.elemSize), sizeof(uint32_t));
    integrateString.append((char*)&(_integrateMeta.fragmentNum), sizeof(uint32_t));
    integrateString.append((char*)&(_integrateMeta.centroidNum), sizeof(uint32_t));
    integrateString.append((char*)_integrateMatrix, _integrateMatrixSize);
}

bool CentroidResource::setValueInRoughMatrix(size_t level, size_t centroidIndex, const void* value)
{
    size_t index = 0;
    size_t ratio = 1;
    for (size_t l = 0; l < level; l++) {
        index += _roughMeta.centroidNums[l] * ratio;
        ratio *= _roughMeta.centroidNums[l];
    }
    index += centroidIndex;
    index *= _roughMeta.elemSize;
    memcpy(_roughMatrix + index, value, _roughMeta.elemSize);
    return true;
}

bool CentroidResource::setValueInIntegrateMatrix(size_t fragmentIndex, size_t centroidIndex, const void* value)
{
    size_t index = (fragmentIndex * _integrateMeta.centroidNum + centroidIndex) * _integrateMeta.elemSize;
    memcpy(_integrateMatrix + index, value, _integrateMeta.elemSize);
    return true;
}

bool CentroidResource::parseRoughContent(char *pBase, size_t fileLength)
{
    if (fileLength < 4 * sizeof(uint32_t)) {
        LOG_ERROR("rough matrix content error length[%ld]", fileLength);
        return false;
    }
    char* buf = pBase;
    size_t offset = 0;
    _roughMeta.magic = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    if (_roughMeta.magic != 0) {
        return false;
    }
    _roughMeta.elemSize = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    _roughMeta.levelCnt = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    _roughMeta.centroidNums.clear();
    size_t totalCentroidNum = 0;
    size_t ratio = 1;
    for (uint32_t i = 0; i < _roughMeta.levelCnt; ++i) {
        uint32_t centroidNum = *((uint32_t *)(buf + offset));
        ratio *= centroidNum;
        totalCentroidNum += ratio;
        _roughMeta.centroidNums.push_back(centroidNum);
        offset += sizeof(uint32_t);
    }
    size_t rightFileLength = offset + totalCentroidNum * _roughMeta.elemSize;
    if (fileLength != rightFileLength)
    {
        LOG_ERROR("rough matrix length[%lu] is not equal rightlength[%lu]", 
                    fileLength, rightFileLength);
        return false;
    }

    _roughMatrix = buf + offset;
    _roughMatrixSize = rightFileLength - offset;
    return true;
}

bool CentroidResource::parseIntegrateContent(char *pBase, size_t fileLength)
{
    if (fileLength < 4 * sizeof(uint32_t)) {
        LOG_ERROR("integrate matrix content error length[%ld]", fileLength);
        return false;
    }
    char* buf = pBase;
    size_t offset = 0;
    _integrateMeta.magic = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    if (_integrateMeta.magic != 0) {
        return false;
    }
    _integrateMeta.elemSize = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    _integrateMeta.fragmentNum = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    _integrateMeta.centroidNum = *((uint32_t *)(buf + offset));
    offset += sizeof(uint32_t);
    size_t rightFileLength = offset + _integrateMeta.centroidNum 
        * _integrateMeta.fragmentNum * _integrateMeta.elemSize;
    if (fileLength != rightFileLength)
    {
        LOG_ERROR("integrate matrix length[%lu] is not equal rightlength[%lu]", 
               fileLength, rightFileLength);
        return false;
    }

    _integrateMatrix = buf + offset;
    _integrateMatrixSize = rightFileLength - offset;
    return true;
}

bool CentroidResource::DumpToFile(const string rough_name)
{
    string tmp;
    if(!rough_name.empty())
    {
        FILE *fp = fopen(rough_name.c_str(), "wb");
        if (!fp) {
            LOG_ERROR("Fopen file [%s] with wb failed:[%s]", rough_name.c_str(), strerror(errno));
            return false;
        }
        dumpRoughMatrix(tmp);

        size_t cnt = fwrite(tmp.c_str(), 1, tmp.size(), fp);
        if (cnt != tmp.size()) {
            LOG_ERROR("Write file header to file [%s] fail", rough_name.c_str());
            fclose(fp);
            return false;
        }
        fclose(fp);
    }

    return true;
}

bool CentroidResource::DumpToFile(const string rough_name, const string& interagte_name)
{
    string tmp;
    if(!rough_name.empty())
    {
        FILE *fp = fopen(rough_name.c_str(), "wb");
        if (!fp) {
            LOG_ERROR("Fopen file [%s] with wb failed:[%s]", rough_name.c_str(), strerror(errno));
            return false;
        }    
        dumpRoughMatrix(tmp);
        
        size_t cnt = fwrite(tmp.c_str(), 1, tmp.size(), fp);
        if (cnt != tmp.size()) {
            LOG_ERROR("Write file header to file [%s] fail", rough_name.c_str());
            fclose(fp);
            return false;
        }
        fclose(fp);   
    }
    
    if(!interagte_name.empty())
    {
        tmp.clear();
        FILE *fp = fopen(interagte_name.c_str(), "wb");
        if (!fp) {
            LOG_ERROR("Fopen file [%s] with wb failed:[%s]", interagte_name.c_str(), strerror(errno));
            return false;
        }    
        dumpIntegrateMatrix(tmp);
        
        size_t cnt = fwrite(tmp.c_str(), 1, tmp.size(), fp);
        if (cnt != tmp.size()) {
            LOG_ERROR("Write file header to file [%s] fail", interagte_name.c_str());
            fclose(fp);
            return false;
        }
        fclose(fp);   
    }
    
    return true;
}

} // namespace mercury
