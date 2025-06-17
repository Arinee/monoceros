#include <cstring>
#include <iostream>
#include "src/core/utils/string_util.h"
#include "common.h"

namespace mercury { namespace core {

std::string DataToStr(const void* data) {
    std::vector<float> float_vec;
    float_vec.resize(256);
    memcpy(float_vec.data(), data, 256 * sizeof(float));
    return StringUtil::vectorToStr(float_vec);
};

std::string DataToStr(const void* data, size_t dim) {
    std::vector<float> float_vec;
    float_vec.resize(dim);
    memcpy(float_vec.data(), data, dim * sizeof(float));
    return StringUtil::vectorToStr(float_vec);
};

std::string DataToStr(const std::vector<const void*>& datas) {
    std::string result;
    for (auto& data : datas) {
        result += DataToStr(data) + ";";
    }

    return result;
}

}}
