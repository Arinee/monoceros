#ifndef __EXAMPLES_PQ_HELPER__
#define __EXAMPLES_PQ_HELPER__

#include "common/pq_codebook.h"
#include "framework/utility/mmap_file.h"
#include <iostream>
#include <assert.h>
#include <vector>

namespace mercury {

bool preparePQCodebook(const mercury::IndexMeta& indexMeta,
                       const std::string& roughMatrix,
                       const std::string& integrateMatrix,
                       PQCodebook::Pointer &pqCodebook)
{
    // read resource from file
    mercury::MMapFile roughFile, integrateFile;
    roughFile.open(roughMatrix.c_str(), true);
    integrateFile.open(integrateMatrix.c_str(), true);
    if (!roughFile.isValid() || !integrateFile.isValid()) {
        std::cerr << "Faild to open resource file" << std::endl;
        return false;
    }
    const char* rough = (char*)roughFile.region() + 4;
    const char* integrate = (char*)integrateFile.region() + 4;

    // Prepare codebook 
    const uint32_t roughCentroidNum = *(uint32_t*)rough;
    rough += 4;
    const uint32_t roughDimension = *(uint32_t*)rough;
    rough += 4;
    const uint32_t integrateCentroidNum = *(uint32_t*)integrate;
    integrate += 4;
    const uint32_t fragmentNum = *(uint32_t*)integrate;
    integrate += 4;
    const uint16_t subDimension = *(uint16_t*)integrate;
    integrate += 4;
    
    std::cout << roughCentroidNum << "|" << roughDimension
        << "|" << integrateCentroidNum
        << "|" << fragmentNum
        << "|" << subDimension
        << std::endl;
    pqCodebook.reset(new PQCodebook(indexMeta, 
                                    roughCentroidNum, 
                                    integrateCentroidNum, 
                                    fragmentNum));
    assert(pqCodebook.get() != nullptr);

    // special case for odps codebook: need to tranform from int16 to float
    size_t roughSize = roughDimension * sizeof(int16_t);
    for (uint32_t i = 0; i < roughCentroidNum; ++i) {
        std::vector<float> val(roughDimension);
        for (size_t k = 0; k < roughDimension; ++k) {
            val[k] = (float)*(int16_t*)(rough + i * roughSize + k * sizeof(int16_t));
        }
        size_t sizeInBytes = val.size()*sizeof(float);
        if (! pqCodebook->appendRoughCentroid(val.data(), sizeInBytes)) {
            std::cerr << "pqCodebook emplace rough centroid error" << std::endl;
            return false;
        }
    }
    size_t integrateSize = subDimension * sizeof(float);
    for (uint32_t i = 0; i < fragmentNum; ++i) {
        for (uint32_t k = 0; k < integrateCentroidNum; ++k) {
            std::vector<float> val(subDimension);
            memcpy(val.data(), 
                    integrate + (i * integrateCentroidNum + k) * integrateSize, 
                    val.size() * sizeof(float));
            if (! pqCodebook->appendIntegrateCentroid(i, val.data(), integrateSize)) {
                std::cerr << "pqCodebook emplace integrate error" << std::endl;
                return false;
            }
        }
    }
    return true;
}


} // namespace mercury

#endif // __EXAMPLES_PQ_HELPER__
