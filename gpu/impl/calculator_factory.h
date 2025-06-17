/*********************************************************************
 * $Author: lingxiao.yaolx $
 *
 * $LastChangedBy: lingxiao.yaolx $
 *
 * $LastChangedDate: 2018-06-12 14:49 $
 *
 * $Id: calculator_factory.h 2018-06-12 14:49 lingxiao.yaolx $
 *
 ********************************************************************/

#ifndef IMPL_CALCULATOR_FACTORY_H_
#define IMPL_CALCULATOR_FACTORY_H_
#include "aitheta/index_meta.h"
#include "aitheta/index_distance.h"
#include "index_distance.cuh"

namespace proxima { namespace gpu {

class CalculatorFactory
{ 
public:
    CalculatorFactory(aitheta::IndexMeta &meta) : _meta(meta) {
    }
    ~CalculatorFactory() {
    }
    Calculator *createCal() {
        aitheta::IndexDistance::Methods method = _meta.method();
        Calculator *cal = NULL;
        switch (method) {
        case aitheta::IndexDistance::kMethodFloatSquaredEuclidean:
            cal = new GPUFloatSquaredEuclideanCalculator();
            break;
        case aitheta::IndexDistance::kMethodFloatInnerProduct:
            cal = new GPUFloatInnerProductCalculator();
            break;
        default:
            break;
        }
        if (cal == NULL) {
            if (_meta.type() == aitheta::VectorHolder::kTypeFloat) {
                cal = new GPUFloatSquaredEuclideanCalculator();
            }
        }
        return cal;
    }
private:
    aitheta::IndexMeta _meta;
};

} }

#endif //IMPL_CALCULATOR_FACTORY_H_
