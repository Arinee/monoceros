/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_plugin.h
 *   \author   Hechong.xyf
 *   \date     Apr 2018
 *   \version  1.0.0
 *   \brief    Interface of Mercury Index Plugin
 */

#ifndef __MERCURY_INDEX_PLUGIN_H__
#define __MERCURY_INDEX_PLUGIN_H__

#include <string>
#include <vector>

namespace mercury {

/*! Index Plugin
 */
class IndexPlugin
{
public:
    //! Constructor
    IndexPlugin(void) : _handle(nullptr) {}

    //! Constructor
    IndexPlugin(IndexPlugin &&plugin) : _handle(plugin._handle)
    {
        plugin._handle = nullptr;
    }

    //! Constructor
    explicit IndexPlugin(const std::string &path) : _handle(nullptr)
    {
        this->load(path);
    }

    //! Destructor
    ~IndexPlugin(void) {}

    //! Test if the plugin is valid
    bool isValid(void) const
    {
        return (!!_handle);
    }

    //! Retrieve the handle
    void *handle(void) const
    {
        return _handle;
    }

    //! Load the library path
    bool load(const std::string &path);

    //! Load the library path
    bool load(const std::string &path, std::string *err);

    //! Unload plugin
    void unload(void);

private:
    //! Disable them
    IndexPlugin(const IndexPlugin &) = delete;
    IndexPlugin &operator=(const IndexPlugin &) = delete;

    //! Members
    void *_handle;
};

/*! Index Plugin Broker
 */
class IndexPluginBroker
{
public:
    //! Constructor
    IndexPluginBroker(void) : _plugins() {}

    //! Constructor
    IndexPluginBroker(IndexPluginBroker &&broker)
        : _plugins(std::move(broker._plugins))
    {
    }

    //! Destructor
    ~IndexPluginBroker(void) {}

    //! Emplace a plugin
    bool emplace(IndexPlugin &&plugin);

    //! Emplace a plugin via library path
    bool emplace(const std::string &path)
    {
        return this->emplace(IndexPlugin(path));
    }

    //! Load plugins into broker (with pattern)
    bool load(const std::string &pattern);

    //! Retrieve count of plugins in broker
    size_t count(void) const
    {
        return _plugins.size();
    }

private:
    //! Disable them
    IndexPluginBroker(const IndexPluginBroker &) = delete;
    IndexPluginBroker &operator=(const IndexPluginBroker &) = delete;

    //! Members
    std::vector<IndexPlugin> _plugins;
};

} // namespace mercury

#endif // __MERCURY_INDEX_PLUGIN_H__
