#pragma once

namespace mercury { namespace core {
        std::string DataToStr(const void* data);
        std::string DataToStr(const void* data, size_t dim);
        std::string DataToStr(const std::vector<const void*>& data);
}}
