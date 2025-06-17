#include <cstring>
#include "src/core/utils/string_util.h"
#include "common.h"

namespace mercury { namespace redindex {
using namespace mercury::core;

std::string DataToStr(const void* data) {
    std::vector<float> float_vec;
    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));
    return StringUtil::vectorToStr(float_vec);
};

}}
