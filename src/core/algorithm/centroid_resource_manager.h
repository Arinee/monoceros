#pragma once

#include "centroid_resource.h"

MERCURY_NAMESPACE_BEGIN(core);

class CentroidResourceManager {
public:
    CentroidResourceManager() {}
    ~CentroidResourceManager() {}

    CentroidResource& GetCentroidResource(gindex_t group_index);
    const CentroidResource& GetCentroidResource(gindex_t group_index) const;
    // 从rough_base load，该rough_base包含多个centroid_resource对应的rough_matrix
    int LoadRough(const void *rough_base, size_t rough_len);
    int DumpRoughMatrix(std::string &rough_string) const;
    // manager 管理group_num个centroid_resource
    //int Create(gindex_t group_num);
    int AddCentroidResource(const CentroidResource&& centroid_resource);
    //根据group偏移，计算在coarseindex的slotindex中的位置
    off_t GetSlotIndex(gindex_t group_index, uint32_t label) const;
    gindex_t GetGroupIndex(uint32_t slot_index) const;
    size_t GetTotalCentroidsNum() const;
    size_t GetCentroidsNum(gindex_t group_index) const;

private:
    std::vector<CentroidResource> centroid_resources_;
    //每个group 0号label的起始offset
    std::vector<off_t> group_starts_;
};

MERCURY_NAMESPACE_END(core);
