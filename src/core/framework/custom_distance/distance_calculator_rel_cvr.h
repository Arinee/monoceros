#ifndef __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_REL_CVR_H__
#define __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_REL_CVR_H__

#include "distance_calculator_base.h"

MERCURY_NAMESPACE_BEGIN(core);

class RelCvrCalculator : public BaseCustomCalculator {
public:
    RelCvrCalculator(size_t elemSize, size_t dimension, IndexDistance::Measure measure, uint32_t part_dimension)
        : BaseCustomCalculator(elemSize, dimension, measure, part_dimension) {}

    float CustomScore(const void* query, const void* goods, size_t query_elem_size) override {
        uint32_t query_dimension = query_elem_size / sizeof(float);
        uint32_t query_part_num = query_dimension / part_dimension_;
        uint32_t goods_part_num = dimension_ / part_dimension_;
        size_t part_size = part_dimension_ * sizeof(float);
        float result = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);
        for (uint32_t i = 0; i < query_part_num; i++) {
            float min = std::numeric_limits<float>::max();
            for (uint32_t j = 0; j < goods_part_num; j++) {
                float score = measure_(query_vec + i * part_size, goods_vec + j * part_size, part_size);
                if (score < min) {
                    min = score;
                }
            }
            result += min;
        }
        return result;
    }
    virtual float CustomRankScore(const void* query, const void* goods, size_t /* query_elem_size */) override {
        size_t part_size = part_dimension_ * sizeof(float);
        float result = 0.0;
        const char* query_vec = reinterpret_cast<const char*>(query);
        const char* goods_vec = reinterpret_cast<const char*>(goods);
        result = measure_(query_vec, goods_vec, part_size);
        return result;
    };
};

MERCURY_NAMESPACE_END(core);

#endif  // __MERCURY_CORE_FRAMEWORK_DISTANCE_CALCULATOR_REL_INF_H__