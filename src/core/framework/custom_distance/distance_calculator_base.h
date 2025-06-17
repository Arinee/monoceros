#ifndef __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_BASE_H__
#define __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_BASE_H__

#include "src/core/common/common.h"
#include "src/core/framework/index_distance.h"
#include "src/core/framework/index_framework.h"

MERCURY_NAMESPACE_BEGIN(core);

enum CustomMethods {
    Default = 0,
    Permutation = 1,  // 排列计算for掘阅
    CtrCvr = 2,       // ctr, cvr交叉计算
    Simple = 3,       // 朴素计算向量点击
    Mobius = 4,       // for 猫头鹰
    RelCvr = 5        // Permutation + cvr交叉

};

class BaseCustomCalculator {
public:
    typedef std::shared_ptr<BaseCustomCalculator> Pointer;

    BaseCustomCalculator(size_t elemSize, size_t dimension, IndexDistance::Measure measure, uint32_t part_dimension)
        : elem_size_(elemSize), dimension_(dimension), measure_(measure), part_dimension_(part_dimension) {}

    virtual ~BaseCustomCalculator() {}

    virtual float CustomScore(const void* query, const void* goods, size_t query_elem_size) = 0;
    virtual float CustomRankScore(const void* query, const void* goods, size_t query_elem_size) { return 0.0; }

protected:
    size_t elem_size_;
    size_t dimension_;
    IndexDistance::Measure measure_;
    uint32_t part_dimension_;
};

MERCURY_NAMESPACE_END(core);

#endif  // __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_BASE_H__