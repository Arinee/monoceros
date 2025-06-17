#ifndef NOTE_AGE_UTIL_H_
#define NOTE_AGE_UTIL_H_

#include "src/core/common/common.h"
#include "src/core/framework/index_logger.h"
#include "putil/src/putil/StringUtil.h"
#include <string>

MERCURY_NAMESPACE_BEGIN(core);
class NoteUtil
{
public:
    // get note create time from noteid
    static bool NoteIdToCreateTimeS(const std::string &note_id, uint32_t &create_timestamp_s){
        if (note_id.length() != 24) {
            LOG_ERROR("note id length != 24: %s", note_id.c_str());
            return false;
        }
        const std::string& hex_num = note_id.substr(0, 8);
        if (putil::StringUtil::hexStrToUint32(hex_num.data(), create_timestamp_s)) {
            return true;
        }
        LOG_ERROR("hexStrToUint32 failed! note_id: %s", note_id.c_str());
        return false;
    }
};

MERCURY_NAMESPACE_END(core);
#endif //NOTE_AGE_UTIL_H_
