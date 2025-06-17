/**
 *   Copyright (C) The Software Authors. All rights reserved.

 *   \file     index_params.cc
 *   \author   qiuming
 *   \date     Jan 2019
 *   \version  1.0.0
 *   \brief    Index Parameters
 */

#include "index_params.h"
#include <cstring>

//! Global environ variable
extern char **environ;

MERCURY_NAMESPACE_BEGIN(core);

void IndexParams::updateFromEnvironment(void)
{
    // Dump all environ string
    for (size_t i = 0; environ[i]; ++i) {
        const char *env = environ[i];
        const char *p = std::strchr(env, '=');
        if (p) {
            std::string key("ENV.");
            key.append(env, p - env);
            this->set(std::move(key), std::string(p + 1));
        }
    }
}

//! Empty parameters
const IndexParams EmptyParams;

MERCURY_NAMESPACE_END(core);
