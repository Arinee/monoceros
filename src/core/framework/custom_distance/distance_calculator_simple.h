//
// Created by rugeng on 2020/7/28.
//

#ifndef __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_SIMPLE_H
#define __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_SIMPLE_H
#include <vector>

#include "distance_calculator_base.h"

MERCURY_NAMESPACE_BEGIN(core);

class SimpleCalculator : public BaseCustomCalculator {
public:
    SimpleCalculator(size_t elemSize, size_t dimension, IndexDistance::Measure measure, uint32_t part_dimension)
        : BaseCustomCalculator(elemSize, dimension, measure, part_dimension) {}

    float CustomScore(const void* query, const void* goods, size_t query_elem_size) override {
        float result = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);
        result = measure_(query_vec, goods_vec, elem_size_);
        return result;
    }
};

MERCURY_NAMESPACE_END(core);
#endif  //__MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_SIMPLE_H
