/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_plugin.cc
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Implementation of Mercury Index Plugin
 */

#include "index_plugin.h"
#include "index_logger.h"
#include <dlfcn.h>
#include <glob.h>

namespace mercury {

bool IndexPlugin::load(const std::string &path)
{
    if (_handle) {
        return false;
    }
    _handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!_handle) {
        LOG_INFO("Load so %s: %s", path.c_str(), dlerror());
    } else  {
        LOG_INFO("Load so %s: success", path.c_str());
    }
    return (!!_handle);
}

bool IndexPlugin::load(const std::string &path, std::string *err)
{
    if (_handle) {
        *err = "plugin loaded";
        return false;
    }
    _handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    *err = dlerror();
    if (!_handle) {
        LOG_INFO("Load so %s: %s", path.c_str(), dlerror());
    } else  {
        LOG_INFO("Load so %s: success", path.c_str());
    }
    return (!!_handle);
}

void IndexPlugin::unload(void)
{
    if (_handle) {
        dlclose(_handle);
        _handle = nullptr;
    }
}

bool IndexPluginBroker::emplace(IndexPlugin &&plugin)
{
    if (!plugin.isValid()) {
        return false;
    }
    for (auto iter = _plugins.begin(); iter != _plugins.end(); ++iter) {
        if (iter->handle() == plugin.handle()) {
            plugin.unload();
            return true;
        }
    }
    _plugins.push_back(std::move(plugin));
    return true;
}

bool IndexPluginBroker::load(const std::string &pattern)
{
    glob_t result;
    if (glob(pattern.c_str(), GLOB_TILDE, NULL, &result) != 0) {
        return false;
    }
    size_t orig = this->count();
    for (size_t i = 0; i < result.gl_pathc; ++i) {
        this->emplace(std::string(result.gl_pathv[i]));
    }
    globfree(&result);
    return (this->count() > orig);
}

} // namespace mercury
