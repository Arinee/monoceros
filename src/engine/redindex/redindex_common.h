/// Copyright (c) 2019, xiaohongshu Inc. All rights reserved.
/// Author: kailuo <kailuo@xiaohongshu.com>
/// Created: 2019-11-25 15:37

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include "src/core/framework/index_params.h"

namespace mercury { namespace redindex {

typedef int32_t RedIndexDocid;
constexpr RedIndexDocid INVALID_REDINDEX_DOC_ID = -1;

constexpr auto TrainDataPath("TrainDataPath");
constexpr auto DataType("DataType");
constexpr auto Method("Method");
constexpr auto Dimension("Dimension");
constexpr auto PqFragmentCnt("PqFragmentCnt");
constexpr auto PqCentroidNum("PqCentroidNum");
constexpr auto IndexType("IndexType");
constexpr auto GroupIvfVisitLimit("GroupIvfVisitLimit");

constexpr auto IvfCoarseScanRatio("IvfCoarseScanRatio");
constexpr auto IvfPQScanNum("IvfPQScanNum");
constexpr auto SCHEMA_BUILDER_MEMORY_QUOTA("schema.builder.memory.quota");
constexpr auto SCHEMA_MERGER_MEMORY_QUOTA("schema.merger.memory.quota");

using SchemaParams = std::map<std::string, std::string>;

bool SchemaToIndexParam(const SchemaParams& schema, mercury::core::IndexParams& index_params);
}}
