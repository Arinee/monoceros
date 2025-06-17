#ifndef __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_MOBIUS_H__
#define __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_MOBIUS_H__

#include "distance_calculator_base.h"

MERCURY_NAMESPACE_BEGIN(core);

class MobiusCalculator : public BaseCustomCalculator {
public:
    MobiusCalculator(size_t elemSize, size_t dimension, IndexDistance::Measure measure, uint32_t part_dimension)
        : BaseCustomCalculator(elemSize, dimension, measure, part_dimension) {}

    float CustomScore(const void* query, const void* goods, size_t /* query_elem_size */) override {
        float result = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);
        // query_vec is twice the length of goods_vec,cut the query_vec into two parts
        // the first part is used for searcher,the second part is used for merger
        result = measure_(query_vec, goods_vec, elem_size_);
        return result;
    }
    float CustomRankScore(const void* query, const void* goods, size_t /* query_elem_size */) override {
        float result = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);

        uint32_t bias = part_dimension_ * sizeof(float);
        result = measure_(query_vec + bias, goods_vec, elem_size_);
        return result;
    }
};

MERCURY_NAMESPACE_END(core);

#endif  // __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_MOBIUS_H__
