#ifndef __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_CTCVR_H__
#define __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_CTCVR_H__

#include "distance_calculator_base.h"

MERCURY_NAMESPACE_BEGIN(core);

class CtcvrCalculator : public BaseCustomCalculator {
public:
    CtcvrCalculator(size_t elemSize, size_t dimension, IndexDistance::Measure measure, uint32_t part_dimension)
        : BaseCustomCalculator(elemSize, dimension, measure, part_dimension) {}

    float CustomScore(const void* query, const void* goods, size_t /*query_elem_size*/) override {
        size_t part_size = part_dimension_ * sizeof(float);

        float ctr_score = 0.0;
        float cvr_score = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);

        ctr_score = measure_(query_vec, goods_vec, part_size);
        cvr_score = measure_(query_vec + part_size, goods_vec + part_size, part_size);

        return ctr_score * cvr_score;
    }
};

MERCURY_NAMESPACE_END(core);

#endif  // __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_CTCVR_H__
