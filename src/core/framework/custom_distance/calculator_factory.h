#ifndef __MERCURY_CORE_CALCULATOR_FACTORY_H__
#define __MERCURY_CORE_CALCULATOR_FACTORY_H__

#include <functional>
#include <memory>

#include "distance_calculator_base.h"
#include "distance_calculator_ctcvr.h"
#include "distance_calculator_mobius.h"
#include "distance_calculator_permutation.h"
#include "distance_calculator_rel_cvr.h"
#include "distance_calculator_simple.h"

MERCURY_NAMESPACE_BEGIN(core);

class CalculatorFactory {
public:
    CalculatorFactory() : part_dimension_(0) {}
    ~CalculatorFactory() {
    }

    int Init(IndexMeta index_meta) {
        index_meta_ = index_meta;
        return 0;
    }

    void SetMethod(uint32_t part_dimension, CustomMethods custom_method) {
        part_dimension_ = part_dimension;
        custom_method_ = custom_method;
    }

    IndexDistance::Measure Create() {
        // if you want to add a new distance calculator_, edit it after the last "else if"
        if (custom_method_ == CustomMethods::Default) {
            return index_meta_.measure();
        } else if (custom_method_ == CustomMethods::Permutation) {
            calculator_.reset(new PermutationCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else if (custom_method_ == CustomMethods::CtrCvr) {
            calculator_.reset(new CtcvrCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else if (custom_method_ == CustomMethods::Simple) {
            calculator_.reset(new SimpleCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else if (custom_method_ == CustomMethods::Mobius) {
            calculator_.reset(new MobiusCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else if (custom_method_ == CustomMethods::RelCvr) {
            calculator_.reset(new RelCvrCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else {
            LOG_ERROR("Unsupport Custom Method");
        }

        IndexDistance::Measure func;
        func = std::bind(&BaseCustomCalculator::CustomScore, calculator_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        return func;
    }

    IndexDistance::Measure CreateRank() {
        // if you want to add a new rank distance calculator_, edit it after the last "else if"
        if (custom_method_ == CustomMethods::Default) {
            return index_meta_.measure();
        } else if (custom_method_ == CustomMethods::Mobius) {
            calculator_.reset(new MobiusCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else if (custom_method_ == CustomMethods::RelCvr) {
            calculator_.reset(new RelCvrCalculator(index_meta_.sizeofElement(), index_meta_.dimension(), index_meta_.measure(), part_dimension_));
        } else {
            LOG_ERROR("Unsupport CustomRank Method");
        }

        IndexDistance::Measure func;
        func = std::bind(&BaseCustomCalculator::CustomRankScore, calculator_, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        return func;
    }

    IndexDistance::Measure Create(GeneralSearchContext* context) const {
        return IndexDistance::EmbodyMeasure(context->getSearchMethod());
    }

private:
    IndexMeta index_meta_;
    uint32_t part_dimension_;
    CustomMethods custom_method_;
    BaseCustomCalculator::Pointer calculator_;
};

MERCURY_NAMESPACE_END(core);

#endif  //__MERCURY_CORE_CALCULATOR_FACTORY_H__