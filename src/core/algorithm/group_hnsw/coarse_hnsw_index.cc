#include <functional>
#include "coarse_hnsw_index.h"

MERCURY_NAMESPACE_BEGIN(core);

CoarseHnswIndex::CoarseHnswIndex() : base_docid_(0) {
    p_base_ = nullptr;
    p_header_ = nullptr;
}

CoarseHnswIndex::~CoarseHnswIndex() {
    p_base_ = nullptr;
    p_header_ = nullptr;
}

bool CoarseHnswIndex::create(void *pBase, size_t capacity, uint32_t group_num)
{
    assert(pBase != nullptr);

    p_base_ = reinterpret_cast<char *>(pBase);
    memset(p_base_, 0, capacity);

    p_header_ = reinterpret_cast<Header *>(p_base_);
    p_header_->group_num = group_num;
    p_header_->capacity = capacity;

    return true;
}

int CoarseHnswIndex::setGroupMeta(uint64_t group_offset, uint64_t group_end_offset, uint32_t group_doc_total_num, bool is_hnsw_group) {
    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(p_base_ + group_offset);
    pCurGroup->group_capacity = group_end_offset - group_offset;
    pCurGroup->doc_total_num = group_doc_total_num;
    pCurGroup->is_hnsw = is_hnsw_group;
    pCurGroup->doc_cur_num = 0;
    pCurGroup->entry_point = std::numeric_limits<docid_t>::max();
    if (is_hnsw_group) {
        pCurGroup->neighbor_cur_offset = sizeof(GroupHeader) + group_doc_total_num * sizeof(uint64_t);
        if (pCurGroup->neighbor_cur_offset > pCurGroup->group_capacity) {
            LOG_ERROR("Memory OVERFLOW!!! setGroupMeta");
            return -1;
        }
    } else {
        pCurGroup->neighbor_cur_offset = sizeof(GroupHeader) + group_doc_total_num * sizeof(docid_t);
    }
    return 0;
}

void CoarseHnswIndex::setLevelOffset(uint32_t max_level, uint64_t upper_neighbor_cnt) {
    
    upper_neighbor_cnt_ = upper_neighbor_cnt;
    base_neighbor_cnt_ = upper_neighbor_cnt_ * 2;

    //last element record the full length for node mapping to _maxLevel
    level_offset_.resize(max_level + 2);
    level_offset_.at(0) = sizeof(docid_t);// offset of level 0
    level_offset_.at(1) = level_offset_.at(0) + sizeof(CoarseHnswIndex::NeighborListHeader) + base_neighbor_cnt_ * sizeof(docid_t);// offset of level 1
    for (uint32_t i = 2; i < (max_level + 2); ++i) {
        level_offset_.at(i) = level_offset_.at(i - 1) + sizeof(CoarseHnswIndex::NeighborListHeader) + upper_neighbor_cnt_ * sizeof(docid_t);
    }
}

void CoarseHnswIndex::setCandidateNums(uint32_t max_level, uint64_t ef_construction) {
    level_topks_.resize(max_level + 1);  
    level_topks_.at(0).reset(std::max(base_neighbor_cnt_, ef_construction));
    for (uint32_t i = 1; i < max_level + 1; ++i) {
        level_topks_.at(i).reset(std::max(upper_neighbor_cnt_, ef_construction));
    }
}

void CoarseHnswIndex::setScorer(uint32_t part_dimension, CustomMethods custom_method) {
    calculator_factory_.Init(*index_meta_);
    calculator_factory_.SetMethod(part_dimension, custom_method);
    measure_ = calculator_factory_.Create();
}

void CoarseHnswIndex::addBruteGroupDoc(uint64_t group_offset, docid_t doc_id) {
    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(p_base_ + group_offset);
    docid_t *doc_id_base = reinterpret_cast<docid_t *>(pCurGroup + 1);
    uint32_t doc_cur_num = pCurGroup->doc_cur_num;
    doc_id_base[doc_cur_num] = doc_id;
    pCurGroup->doc_cur_num += 1;
}

int64_t CoarseHnswIndex::addHnswGroupDocMeta(uint64_t group_offset, docid_t doc_id, uint32_t doc_max_layer) {

    char* group_base = p_base_ + group_offset;
    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);
    uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(pCurGroup + 1);

    uint32_t doc_cur_num = pCurGroup->doc_cur_num;
    doc_offset_base[doc_cur_num] = pCurGroup->neighbor_cur_offset;
    docid_t *global_doc_id_base = reinterpret_cast<docid_t *>(group_base + doc_offset_base[doc_cur_num]);
    *global_doc_id_base = doc_id;
    pCurGroup->doc_cur_num += 1;

    // 计算新加入点导致的neighbor offset变化
    pCurGroup->neighbor_cur_offset += level_offset_.at(doc_max_layer + 1);
    if (pCurGroup->neighbor_cur_offset > pCurGroup->group_capacity) {
        LOG_ERROR("Memory will OVERFLOW!!! addHnswGroupDocMeta failed!");
        return -1;
    }
    
    for (size_t i = 0; i <= doc_max_layer; i++) {
        NeighborListHeader *header = getNeighborList(group_base, doc_offset_base, i, doc_cur_num);
        header->neighbor_cnt = 0;
        header->pos = 0;
        header->state_type = 0;
        header->padding = 0xfe;
    }

    return doc_cur_num;
}

int CoarseHnswIndex::addDoc(uint64_t group_offset, docid_t group_doc_id, const void *val, int32_t doc_max_layer)
{
    if (val == nullptr) {
        LOG_ERROR("Input vector can't be nullptr");
        return IndexError_InvalidArgument;
    }

    // 提前计算偏移，减少后续重复计算
    char* group_base = p_base_ + group_offset;
    uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(group_base + sizeof(GroupHeader));
    
    // 读取group头，查看入点
    auto hash_value = std::hash<std::string>{}(std::to_string(group_offset));
    int groupLockId = hash_value & GROUP_LOCK_MASK;
    group_lock_[groupLockId].lock();
    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);
    docid_t entry_point = pCurGroup->entry_point;
    int32_t oldMaxLevel  = static_cast<int32_t>(pCurGroup->cur_max_level);
    uint32_t group_doc_total_num = pCurGroup->doc_total_num;

    // 如果是该group第一个点，设置为入点，并更新当前最高层后返回即可
    if (unlikely(entry_point == std::numeric_limits<docid_t>::max())) {
        pCurGroup->entry_point = group_doc_id;
        pCurGroup->cur_max_level = doc_max_layer;
        group_lock_[groupLockId].unlock();
        return 0;
    }

    group_lock_[groupLockId].unlock();

    //获取与入点之间的信息
    docid_t entryPointGlobalId = getGlobalDocId(group_base, doc_offset_base, entry_point);
    const void *entryPointVal = feature_profile_->getInfo(entryPointGlobalId);
    int32_t level = oldMaxLevel;
    float dist = measure_(val, entryPointVal, index_meta_->sizeofElement());

    //doc_max_layer上面的层、oldMaxLevel下面的层更新入点
    uint64_t compare_cnt = 0;
    for (; level > doc_max_layer; --level) {
        if (updateEntryPoint(group_offset, group_base, doc_offset_base, val, index_meta_->sizeofElement(), level, entry_point, dist, compare_cnt, nullptr) != 0) {
            return IndexError_OutOfRange;
        }
    }

    std::vector<TopkHeap> level_topks = level_topks_;
    //curLayer及下面的层更新入点及邻居信息
    for (; level >= 0; --level) {
        TopkHeap &level_topk_heap = level_topks.at(level);
        //选取邻居并建立连接
        addNeighbors(group_offset, group_base, doc_offset_base, group_doc_id, val, level, entry_point, dist, group_doc_total_num, level_topk_heap);
    }

    for (int32_t i = std::min(doc_max_layer, oldMaxLevel); i >= 0; --i) {
        //更新邻居的邻居
        TopkHeap &level_topk_heap = level_topks.at(i);
        updateNeighborLink(group_offset, group_base, doc_offset_base, group_doc_id, i, level_topk_heap);
    }
    group_lock_[groupLockId].lock();
    int32_t newMaxLevel = pCurGroup->cur_max_level;

    //如果doc_max_layer > newMaxLevel需要更新group头
    if (doc_max_layer > newMaxLevel) {
        pCurGroup->entry_point = group_doc_id;
        pCurGroup->cur_max_level = doc_max_layer;
    }
    group_lock_[groupLockId].unlock();
    return 0;
}

int CoarseHnswIndex::updateEntryPoint(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point,
                                      float &dist, uint64_t &compare_cnt, GeneralSearchContext* context)
{
    while (true) {
        //position base on neighborList, index begin with 1
        NeighborListHeader *header;
        int endPos;
        if (unlikely(context == nullptr)) {
            uint64_t group_and_doc_id_operator = group_offset + entry_point;
            auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
            int lockIdx = hash_value & LOCK_MASK;
            std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
            header = getNeighborList(group_base, doc_offset_base, level, entry_point);
            endPos = static_cast<int>(header->neighbor_cnt) - 1;
        } else {
            header = getNeighborList(group_base, doc_offset_base, level, entry_point);
            endPos = static_cast<int>(header->neighbor_cnt) - 1;
        }
        if (endPos < 0) {
            return 0;
        }
        int beginPos = ((endPos + 1) >= step_) ? (endPos + 1  - step_) : 0;

        while (true) {
            bool findCloser = false;
            docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1);
            for (int i = beginPos; i <= endPos; ++i) {
                docid_t node = curNeighbor[i];
                docid_t nodeGlobalId = getGlobalDocId(group_base, doc_offset_base, node);
                const void * nodePointVal = nullptr;
                if (likely(vector_retriever_.isValid() && !contain_feature_)) {
                    if (!vector_retriever_(base_docid_ + nodeGlobalId, nodePointVal)) {
                        LOG_ERROR("retrieve vector failed. docid:%u", nodeGlobalId);
                        return -1;
                    }
                } else {
                    nodePointVal = feature_profile_->getInfo(nodeGlobalId);
                }

                float curDist = measure_(query_val, nodePointVal, len);
                compare_cnt++;
                if (curDist < dist) {
                    entry_point = node;
                    dist = curDist;
                    findCloser = true;
                }
            }

            if (findCloser) {
                break;
            }
            //can't find any neighbor much closer to current entry point
            endPos = beginPos - 1;
            if (endPos < 0) {
                return 0;
            }
            beginPos = ((endPos + 1) >= step_) ? (endPos + 1 - step_) : 0;
        }
    }
    return 0;
}

void CoarseHnswIndex::RedundantMemClip(std::vector<uint64_t*>& offsets) {
    
    char *cur_pos = p_base_ + *offsets.at(0);
    for (auto &offset : offsets) {
        char *group_base = p_base_ + *offset;
        GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);
        uint64_t group_len = pCurGroup->neighbor_cur_offset;
        memmove(cur_pos, group_base, group_len);
        *offset = cur_pos - p_base_;
        cur_pos += group_len;
    }
    uint64_t total_len = cur_pos - p_base_;
    LOG_INFO("Before RedundantMemClip capacity: %lu", p_header_->capacity);
    p_header_->capacity = total_len;
    LOG_INFO("After RedundantMemClip capacity: %lu", p_header_->capacity);
    return;
}

void CoarseHnswIndex::addNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id, const void *val, 
                                   int32_t level, docid_t &entry_point, float &dist, uint32_t group_doc_total_num, TopkHeap& level_topk_heap)
{
    uint64_t compare_cnt = 0;
    //寻找候选点
    searchNeighbors(group_offset, group_base, doc_offset_base, val, index_meta_->sizeofElement(), level, entry_point, dist, level_topk_heap, compare_cnt, group_doc_total_num, nullptr, 0);
    //探索式选取邻居
    selectNeighbors(group_offset, group_base, doc_offset_base, group_doc_id, level, level_topk_heap);
    
    return;
}

//通过邻居的邻居扩展候选点集合
int CoarseHnswIndex::searchNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point,
                                      float &dist, TopkHeap &topk_heap, uint64_t &compare_cnt, uint32_t group_doc_total_num, GeneralSearchContext* context, int max_scan_num_in_query) {
    int max_scan_num = (max_scan_num_in_query ? max_scan_num_in_query : max_scan_num_);
    VisitList<uint32_t> visit_list;
    visit_list.init(group_doc_total_num, max_scan_num);
    CandidateHeap candidates;
    candidates.reset(max_scan_num);
    visit_list.setVisited(entry_point);
    topk_heap.push(entry_point, dist);
    NeighborListHeader *header;
    int curPos;
    if (context == nullptr) {
        uint64_t group_and_doc_id_operator = group_offset + entry_point;
        auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
        int lockIdx = hash_value & LOCK_MASK;
        std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
        header = getNeighborList(group_base, doc_offset_base, level, entry_point);
        curPos = static_cast<int>(header->neighbor_cnt) - 1;
    } else {
        header = getNeighborList(group_base, doc_offset_base, level, entry_point);
        curPos = static_cast<int>(header->neighbor_cnt) - 1;
    }
    candidates.emplace_push(reinterpret_cast<docid_t *>(header + 1), entry_point, curPos, dist);
    //current direction is closing to the nearest node
    bool closeTo = true;

    while (!candidates.empty()) {
        NodeSearchContext *candiCtx;
        float candiDist;
        candidates.top(candiCtx, candiDist);
        if (topk_heap.full() && candiDist > topk_heap.topValue()) {
            break;
        }

        //if all neighbors of current node has been visited
        int endPos = candiCtx->curPos;
        if (endPos < 0) {
            candidates.pop();
            continue;
        }
        int beginPos = ((endPos + 1) >= step_) ? (endPos + 1  - step_) : 0;

        //if closing to nearest node, we detect the farthest neighbor first 
        if (closeTo) {
            while (true) {
                bool findCloser = false;
                docid_t *curNeighbor = candiCtx->neighborList;
                for (int i = beginPos; i <= endPos; ++i) {
                    docid_t node = curNeighbor[i];
                    if (visit_list.visited(node)) {
                        continue;
                    }

                    visit_list.setVisited(node);
                    if (visit_list.needBreak()) {
                        return 0;
                    }
                    docid_t nodeGlobalId = getGlobalDocId(group_base, doc_offset_base, node);
                    const void * nodePointVal = nullptr;
                    if (likely(vector_retriever_.isValid() && !contain_feature_)) {
                        if (!vector_retriever_(base_docid_ + nodeGlobalId, nodePointVal)) {
                            LOG_ERROR("retrieve vector failed. docid:%u", nodeGlobalId);
                            return -1;
                        }
                    } else {
                        nodePointVal = feature_profile_->getInfo(nodeGlobalId);
                    }

                    float curDist = measure_(query_val, nodePointVal, len);
                    compare_cnt++;
                    if ((!topk_heap.full())  || curDist < topk_heap.topValue()) {
                        topk_heap.push(node, curDist);

                        if (context == nullptr) {
                            uint64_t group_and_doc_id_operator = group_offset + node;
                            auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
                            int lockIdx = hash_value & LOCK_MASK;
                            std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
                            header = getNeighborList(group_base, doc_offset_base, level, node);
                            docid_t * node_neighbor_list = reinterpret_cast<docid_t *>(header + 1);
                            int node_cur_pos = static_cast<int>(header->neighbor_cnt) - 1;
                            candidates.emplace_push(node_neighbor_list, node, node_cur_pos, curDist);
                        } else {
                            header = getNeighborList(group_base, doc_offset_base, level, node);
                            docid_t * node_neighbor_list = reinterpret_cast<docid_t *>(header + 1);
                            int node_cur_pos = static_cast<int>(header->neighbor_cnt) - 1;
                            candidates.emplace_push(node_neighbor_list, node, node_cur_pos, curDist);
                        }

                        if (curDist < dist) {
                            entry_point = node;
                            dist = curDist;
                            findCloser = true;
                        }
                    }
                }
                endPos = beginPos - 1;
                candiCtx->curPos = endPos;
                if (findCloser) {
                    break;
                }
                //第一次出现从一个点的所有邻居中都找不到更近的点的时候，跳出循环，进入else分支，之后不再更新entryPoint
                //can't find any neighbor much closer to current entry point
                //begin to expand detection from current node
                if (endPos < 0) {
                    closeTo = false;
                    // candidates.pop();
                    break;
                }
                beginPos = ((endPos + 1) >= step_) ? (endPos + 1 - step_) : 0;
            }
        } else {
            candidates.pop();

            docid_t *curNeighbor = candiCtx->neighborList;
            for (int i = 0; i <= endPos; ++i) {
                docid_t node = curNeighbor[i];
                if (visit_list.visited(node)) {
                    continue;
                }

                visit_list.setVisited(node);
                if (visit_list.needBreak()) {
                    return 0;
                }
                docid_t nodeGlobalId = getGlobalDocId(group_base, doc_offset_base, node);
                const void * nodePointVal = nullptr;
                if (likely(vector_retriever_.isValid() && !contain_feature_)) {
                    if (!vector_retriever_(base_docid_ + nodeGlobalId, nodePointVal)) {
                        LOG_ERROR("retrieve vector failed. docid:%u", nodeGlobalId);
                        return -1;
                    }
                } else {
                    nodePointVal = feature_profile_->getInfo(nodeGlobalId);
                }

                float curDist = measure_(query_val, nodePointVal, len);
                compare_cnt++;
                if ((!topk_heap.full())  || curDist < topk_heap.topValue()) {
                    topk_heap.push(node, curDist);
                    if (context == nullptr) {
                        uint64_t group_and_doc_id_operator = group_offset + node;
                        auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
                        int lockIdx = hash_value & LOCK_MASK;
                        std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
                        header = getNeighborList(group_base, doc_offset_base, level, node);
                        docid_t * node_neighbor_list = reinterpret_cast<docid_t *>(header + 1);
                        int node_cur_pos = static_cast<int>(header->neighbor_cnt) - 1;
                        candidates.emplace_push(node_neighbor_list, node, node_cur_pos, curDist);
                    } else {
                        header = getNeighborList(group_base, doc_offset_base, level, node);
                        docid_t * node_neighbor_list = reinterpret_cast<docid_t *>(header + 1);
                        int node_cur_pos = static_cast<int>(header->neighbor_cnt) - 1;
                        candidates.emplace_push(node_neighbor_list, node, node_cur_pos, curDist);
                    }
                }
            }//end for
        } //end of if (closeTo)
    }
    return 0;
}

int CoarseHnswIndex::searchZeroNeighbors(char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point, 
                                      float &dist, TopkHeap &topk_heap, uint64_t &compare_cnt, 
                                      VisitList<uint32_t>& visit_list, CandidateHeap& candidates) {
    visit_list.setVisited(entry_point);
    topk_heap.push(entry_point, dist);
    NeighborListHeader *header = getNeighborList(group_base, doc_offset_base, level, entry_point);
    int neighbor_cnt = static_cast<int>(header->neighbor_cnt);
    candidates.emplace_push(reinterpret_cast<docid_t *>(header + 1), 
                            entry_point, neighbor_cnt - 1, dist);
    //current direction is closing to the nearest node
    bool closeTo = true;

    while (!candidates.empty()) {
        NodeSearchContext *candiCtx;
        float candiDist;
        candidates.top(candiCtx, candiDist);
        if (topk_heap.full() && candiDist > topk_heap.topValue()) {
            break;
        }
        if (candiCtx->neighborList == nullptr) {
            header = getNeighborList(group_base, doc_offset_base, level, candiCtx->idx);
            candiCtx->neighborList = reinterpret_cast<docid_t *>(header + 1);
            candiCtx->curPos = static_cast<int>(header->neighbor_cnt) - 1;
        }

        //if all neighbors of current node has been visited
        int endPos = candiCtx->curPos;
        if (endPos < 0) {
            candidates.pop();
            continue;
        }
        int beginPos = ((endPos + 1) >= step_) ? (endPos + 1  - step_) : 0;

        //if closing to nearest node, we detect the farthest neighbor first 
        if (closeTo) {
            while (true) {
                bool findCloser = false;
                docid_t *curNeighbor = candiCtx->neighborList + beginPos;
                for (int i = beginPos; i <= endPos; ++i) {
                    docid_t node = *curNeighbor++;
                    if (visit_list.visited(node)) {
                        continue;
                    }

                    visit_list.setVisited(node);
                    if (visit_list.needBreak()) {
                        candidates.reset();
                        visit_list.reset(visit_list.getMaxDocCnt());
                        return 0;
                    }

                    const void * nodePointVal = nullptr;
                    if (likely(vector_retriever_.isValid() && !contain_feature_)) {
                        if (!vector_retriever_(base_docid_ + node, nodePointVal)) {
                            LOG_ERROR("retrieve vector failed. docid:%u", node);
                            candidates.reset();
                            visit_list.reset(visit_list.getMaxDocCnt());
                            return -1;
                        }
                    } else {
                        nodePointVal = feature_profile_->getInfo(node);
                    }

                    float curDist = measure_(query_val, nodePointVal, len);
                    compare_cnt++;
                    if ((!topk_heap.full())  || curDist < topk_heap.topValue()) {
                        topk_heap.push(node, curDist);
                        //we will get the valid value when node really need to be scanned
                        candidates.emplace_push(nullptr, node, 0, curDist);
                        if (curDist < dist) {
                            entry_point = node;
                            dist = curDist;
                            findCloser = true;
                        }
                    }
                }
                endPos = beginPos - 1;
                candiCtx->curPos = endPos;
                if (findCloser) {
                    break;
                }
                //第一次出现从一个点的所有邻居中都找不到更近的点的时候，跳出循环，进入else分支，之后不再更新entryPoint
                //can't find any neighbor much closer to current entry point
                //begin to expand detection from current node
                if (endPos < 0) {
                    closeTo = false;
                    candidates.pop();
                    break;
                }
                beginPos = ((endPos + 1) >= step_) ? (endPos + 1 - step_) : 0;
            }
        } else {
            candidates.pop();

            docid_t *curNeighbor = candiCtx->neighborList;
            for (int i = 0; i <= endPos; ++i) {
                docid_t node = curNeighbor[i];
                if (visit_list.visited(node)) {
                    continue;
                }

                visit_list.setVisited(node);
                if (visit_list.needBreak()) {
                    candidates.reset();
                    visit_list.reset(visit_list.getMaxDocCnt());
                    return 0;
                }

                const void * nodePointVal = nullptr;
                if (likely(vector_retriever_.isValid() && !contain_feature_)) {
                    if (!vector_retriever_(base_docid_ + node, nodePointVal)) {
                        LOG_ERROR("retrieve vector failed. docid:%u", node);
                        candidates.reset();
                        visit_list.reset(visit_list.getMaxDocCnt());
                        return -1;
                    }
                } else {
                    nodePointVal = feature_profile_->getInfo(node);
                }

                float curDist = measure_(query_val, nodePointVal, len);
                compare_cnt++;
                if ((!topk_heap.full())  || curDist < topk_heap.topValue()) {
                    topk_heap.push(node, curDist);
                    candidates.emplace_push(nullptr, node, 0, curDist);
                }
                
            }//end for
        } //end of if (closeTo)
    }
    candidates.reset();
    visit_list.reset(visit_list.getMaxDocCnt());
    return 0;
}

//探索式算法从候选点集合选择最终邻居并写入，不存在竞争关系，无需加锁
void CoarseHnswIndex::selectNeighbors(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id, int32_t level, TopkHeap &topk_heap)
{
    topk_heap.order();
    docid_t *keyPtr = nullptr;
    float *distPtr = nullptr;
    int size = 0;
    topk_heap.peep(keyPtr, distPtr, size);
    int keep_size = static_cast<int>((level > 0) ? upper_neighbor_cnt_ : base_neighbor_cnt_); 
    NeighborListHeader *header;
    {
        uint64_t group_and_doc_id_operator = group_offset + group_doc_id;
        auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
        int lockIdx = hash_value & LOCK_MASK;
        std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
        header = getNeighborList(group_base, doc_offset_base, level, group_doc_id);
        if (size <= keep_size) {
            //must write memory first, then modify header!
            memcpy(reinterpret_cast<docid_t *>(header + 1), keyPtr, size * sizeof(docid_t));
            __asm__ __volatile__("" ::: "memory");
            header->neighbor_cnt = static_cast<uint8_t>(size);
            header->pos = static_cast<uint8_t>(size - 1);
            header->state_type = NeighborListHeader::ORDERED_STATE;
            return;
        }
    }

    docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1);
    int curSize = 0;
    //topk heap saves dist to speed up updating neighbors' link
    docid_t *curKey = keyPtr;
    float *curDist = distPtr;
    //探索式加入最近邻
    for (int i = 0; i < size; ++i) {
        docid_t curNode = keyPtr[i];
        float curNodeDist = distPtr[i];
        bool good = true;
        docid_t curNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, curNode);
        const void * curNodePointVal = feature_profile_->getInfo(curNodeGlobalId);
        for (int j = 0; j < curSize; ++j) {
            docid_t curNeighborGlobalId = getGlobalDocId(group_base, doc_offset_base, curNeighbor[j]);
            const void * curNeighborPointVal = feature_profile_->getInfo(curNeighborGlobalId);
            float tmpDist = measure_(curNodePointVal, curNeighborPointVal, index_meta_->sizeofElement());
            if (tmpDist < curNodeDist) {
                good = false;
                break;
            }
        }
        if (good) {
            curNeighbor[curSize++] = curNode;
            *curKey++ = curNode;
            *curDist++ = curNodeDist;
            if (curSize >= keep_size) {
                break;
            }
        }
    } //end for

    {
        uint64_t group_and_doc_id_operator = group_offset + group_doc_id;
        auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
        int lockIdx = hash_value & LOCK_MASK;
        std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
        __asm__ __volatile__("" ::: "memory");
        header->neighbor_cnt = static_cast<uint8_t>(curSize);
        header->pos = static_cast<uint8_t>(curSize - 1);
        header->state_type = NeighborListHeader::SELECTED_STATE;
    }
    topk_heap.setSize(curSize);
    return;
}

//更新邻居的邻居信息
void CoarseHnswIndex::updateNeighborLink(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t group_doc_id, int32_t level, TopkHeap& levelTopk)
{
    const docid_t *keyPtr = nullptr;
    const float *distPtr = nullptr;
    int size = 0;
    levelTopk.peep(keyPtr, distPtr, size);
    for (int i = 0; i < size; ++i) {
        updateLink(group_offset, group_base, doc_offset_base, keyPtr[i], group_doc_id, level, distPtr[i]);
    }
    return;
}

//更新某一层邻居的邻居信息，可能存在竞争关系，需要加锁
void CoarseHnswIndex::updateLink(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, docid_t main_node, docid_t link_node, int32_t level, float dist)
{
    docid_t effectiveIdx = -1;
    int rightPos = -1;
    float lastDist = 0;
    float newDist = 0;

    uint64_t group_and_doc_id_operator = group_offset + main_node;
    auto hash_value = std::hash<std::string>{}(std::to_string(group_and_doc_id_operator));
    int lockIdx = hash_value & LOCK_MASK;
    std::lock_guard<std::mutex> lock(offset_lock_[lockIdx]);
    
    NeighborListHeader *header = getNeighborList(group_base, doc_offset_base, level, main_node);
    docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1);
    int listSize = static_cast<int>(header->neighbor_cnt);
    
    //当前点邻居数未达到上限，直接写入返回
    int keep_size = static_cast<int>((level > 0) ? upper_neighbor_cnt_ : base_neighbor_cnt_);
    if (listSize < keep_size) {
        curNeighbor[listSize] = link_node;
        __asm__ __volatile__("" ::: "memory");
        (header->neighbor_cnt)++;
        return;
    }

    float *selected_dist = new (std::nothrow) float[keep_size];
    for (int i = 0; i < keep_size; i++) {
        selected_dist[i] = std::numeric_limits<float>::max();
    }
    int selected_pos = static_cast<int>(header->pos);
    if (header->state_type == NeighborListHeader::ORDERED_STATE) {
        //make ordered state to selected state for neighbor before header->pos
        selected_pos = makeOrdered2Selected(group_base, doc_offset_base, main_node, header, selected_dist);
    }
    selected_pos = selectLeftNeighbor(group_base, doc_offset_base, main_node, header, keep_size, selected_pos, selected_dist);

    //process link_node
    listSize = selected_pos + 1;
    int iterSize = listSize;

    //compare with last neighbor, if neighbor list is full and link_node with a bigger dist, link_node should be skipped
    docid_t mainNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, main_node);
    const void * mainNodePointVal = feature_profile_->getInfo(mainNodeGlobalId);

    if (listSize == keep_size) {
        lastDist = selected_dist[listSize - 1];
        if (lastDist == std::numeric_limits<float>::max()) {
            docid_t neighborNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, curNeighbor[listSize - 1]);
            const void * neighborNodePointVal = feature_profile_->getInfo(neighborNodeGlobalId);
            lastDist = measure_(mainNodePointVal, neighborNodePointVal, index_meta_->sizeofElement());
            selected_dist[listSize - 1] = lastDist;
        }

        if (dist >= lastDist) {
            header->neighbor_cnt = keep_size;
            header->state_type = NeighborListHeader::SELECTED_STATE;
            header->pos = keep_size - 1;
            delete [] selected_dist;
            return;
        }
        iterSize = listSize - 1;
    }
    int effectiveSize = listSize;

    //as neighbors with dist ascending order, find a position to insert link_node 
    //before this position, we only need to think about whether link_node should be selected
    //after this position, we think about whether nodes in neighbor should be kicked out
    docid_t linkNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, link_node);
    const void * linkNodePointVal = feature_profile_->getInfo(linkNodeGlobalId);
    for (int i = 0; i < iterSize; ++i) {
        docid_t node = curNeighbor[i];
        docid_t neighborNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, node);
        const void * neighborNodePointVal = feature_profile_->getInfo(neighborNodeGlobalId);
        float neighborDist = selected_dist[i];
        if (neighborDist == std::numeric_limits<float>::max()) {
            neighborDist = measure_(mainNodePointVal, neighborNodePointVal, index_meta_->sizeofElement());
        }

        if ((rightPos == -1) && (neighborDist > dist)) {
            rightPos = i;
            effectiveIdx = i;
        }

        newDist = measure_(linkNodePointVal, neighborNodePointVal, index_meta_->sizeofElement());
        if (rightPos == -1) {
            //current node can't be selected
            if (newDist < dist) {
                __asm__ __volatile__("" ::: "memory");
                header->neighbor_cnt = selected_pos + 1;
                header->state_type = NeighborListHeader::SELECTED_STATE;
                header->pos = selected_pos;
                delete [] selected_dist;
                return;
            }
        } else {
            //if node in neighbor list is obsolete
            if (newDist < neighborDist) {
                effectiveSize--;
            } else {
                curNeighbor[effectiveIdx] = node;
                effectiveIdx++;
            }
        }
    }

    //to save one time of distance computation for last neighbor
    if (listSize == keep_size) {
        //we can make sure dist < lastDist, as we test it before
        if (rightPos == -1) {
            curNeighbor[listSize - 1] = link_node;
            __asm__ __volatile__("" ::: "memory");
            header->neighbor_cnt = listSize;
            header->state_type = NeighborListHeader::SELECTED_STATE;
            header->pos = selected_pos;
            delete [] selected_dist;
            return;
        } else {
            //if node in neighbor list is obsolete, need to handle the last neighbor
            docid_t node = curNeighbor[listSize - 1];
            docid_t neighborNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, node);
            const void * neighborNodePointVal = feature_profile_->getInfo(neighborNodeGlobalId);
            newDist = measure_(linkNodePointVal, neighborNodePointVal, index_meta_->sizeofElement());
            if (newDist < lastDist) {
                effectiveSize--;
            } else {
                curNeighbor[effectiveIdx] = curNeighbor[listSize - 1];
                effectiveIdx++;
            }
        }
    }

    if (rightPos == -1) {
        if (effectiveSize < keep_size) {
            int newSize = effectiveSize + 1;
            curNeighbor[effectiveSize] = link_node;
            __asm__ __volatile__("" ::: "memory");
            header->neighbor_cnt = newSize;
            header->state_type = NeighborListHeader::SELECTED_STATE;
            header->pos = newSize - 1;
        }
    } else {
        int endPos = (effectiveSize >= keep_size) ? (keep_size - 1): effectiveSize;
        //rightPos using index start with 0, when move memory it should +2
        for (int i = endPos; i >= rightPos + 1; --i) {
            curNeighbor[i] = curNeighbor[i - 1];
        }

        curNeighbor[rightPos] = link_node;
        __asm__ __volatile__("" ::: "memory");
        header->neighbor_cnt = endPos + 1;
        header->state_type = NeighborListHeader::SELECTED_STATE;
        header->pos = endPos;
    }  
    delete [] selected_dist;
    return;
}

int CoarseHnswIndex::makeOrdered2Selected(char* group_base, uint64_t *doc_offset_base, docid_t main_node, NeighborListHeader *header, float *selected_dist)
{
    docid_t mainNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, main_node);
    const void * mainNodePointVal = feature_profile_->getInfo(mainNodeGlobalId);
    docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1);
    int curSize = 0;
    docid_t *curKey = curNeighbor;
    int maxPos = static_cast<int>(header->pos);
    for (int i = 0; i <= maxPos; ++i) {
        docid_t curNode = curNeighbor[i];
        docid_t curNeighborGlobalId = getGlobalDocId(group_base, doc_offset_base, curNode);
        const void * curNeighborPointVal = feature_profile_->getInfo(curNeighborGlobalId);
        float curNodeDist = measure_(mainNodePointVal, curNeighborPointVal, index_meta_->sizeofElement());
        bool good = true;
        for (int j = 0; j < curSize; ++j) {
            docid_t curGlobalId = getGlobalDocId(group_base, doc_offset_base, curNeighbor[j]);
            const void * curPointVal = feature_profile_->getInfo(curGlobalId);
            float tmpDist = measure_(curNeighborPointVal, curPointVal, index_meta_->sizeofElement());
            if (tmpDist < curNodeDist) {
                good = false;
                break;
            }
        }
        
        if (good) {
            *curKey++ = curNode;
            *selected_dist++ = curNodeDist;
            curSize++;
        }
    } //end for

    return curSize - 1;
}

int CoarseHnswIndex::selectLeftNeighbor(char* group_base, uint64_t *doc_offset_base, docid_t main_node, NeighborListHeader *header, int keep_size, int selected_pos, float *selected_dist)
{
    docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1);
    int startPos = static_cast<int>(header->pos) + 1;
    docid_t mainNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, main_node);
    const void * mainNodePointVal = feature_profile_->getInfo(mainNodeGlobalId);
    for (int i = startPos; i < keep_size; ++i) {
        docid_t checkNode = curNeighbor[i];
        docid_t checkNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, checkNode);
        const void *checkNodePointVal = feature_profile_->getInfo(checkNodeGlobalId);
        float checkDist = measure_(mainNodePointVal, checkNodePointVal, index_meta_->sizeofElement());

        int rightPos = -1;
        docid_t effectiveIdx = -1;
        int effectiveSize = selected_pos + 1;
        bool skip = false;
        for (int j = 0; j <= selected_pos; ++j) {
            docid_t neighborNode = curNeighbor[j];
            docid_t neighborNodeGlobalId = getGlobalDocId(group_base, doc_offset_base, neighborNode);
            const void * neighborNodePointVal = feature_profile_->getInfo(neighborNodeGlobalId);

            float neighborDist = selected_dist[j];
            if (selected_dist[j] == std::numeric_limits<float>::max()) {
                neighborDist = measure_(mainNodePointVal, neighborNodePointVal, index_meta_->sizeofElement());
                selected_dist[j] = neighborDist;
            }

            if ((rightPos == -1) && (neighborDist > checkDist)) {
                rightPos = j;
                effectiveIdx = j;
            }

            float newDist = measure_(neighborNodePointVal, checkNodePointVal, index_meta_->sizeofElement());
            if (rightPos == -1) {
                if (newDist < checkDist) {
                    skip = true;
                    break;
                }
            } else {
                if (newDist < neighborDist) {
                    effectiveSize--;
                } else {
                    curNeighbor[effectiveIdx] = neighborNode;
                    selected_dist[effectiveIdx] = neighborDist;
                    effectiveIdx++;
                }
            }
        } //end for seleted neighbors

        if (skip) {
            continue;
        }

        if (rightPos == -1) {
            //check node append to selected list
            curNeighbor[++selected_pos] = checkNode;
        } else {
            for (int k = effectiveSize; k >= rightPos + 1; --k) {
                curNeighbor[k] = curNeighbor[k - 1];
                selected_dist[k] = selected_dist[k - 1];
            }
            curNeighbor[rightPos] = checkNode;
            selected_dist[rightPos] = checkDist;
            selected_pos = effectiveSize;
        }
    }//end for left neighbors

    return selected_pos;
}

int CoarseHnswIndex::load(void *pBase, size_t memory_size) {
    assert(pBase != nullptr);
    p_base_ = (char *)pBase;
    p_header_ = (Header*) pBase;
    if ((size_t)p_header_->capacity != memory_size) {
        std::cerr << "file size in header is not equal to real file size" << std::endl;
        return -1;
    }
    LOG_INFO("p_header_->capacity: %lu", p_header_->capacity);
    return 0;
}

int CoarseHnswIndex::searchHnswNeighbors(uint64_t group_offset, TopkHeap &topk_heap, const void *query_val, size_t len, GeneralSearchContext* context, uint64_t &compare_cnt, int max_scan_num_in_query) {

    // 提前计算偏移，减少后续重复计算
    char* group_base = p_base_ + group_offset;
    uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(group_base + sizeof(GroupHeader));

    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);
    docid_t entry_point = pCurGroup->entry_point;
    int32_t max_level = static_cast<int32_t>(pCurGroup->cur_max_level);
    bool is_hnsw_group = pCurGroup->is_hnsw;
    if (!is_hnsw_group) {
        LOG_ERROR("Not a HNSW group!");
        return -1;
    }
    
    if (unlikely(entry_point == std::numeric_limits<docid_t>::max())) {
        LOG_ERROR("No entry point!");
        return -1;
    }
    docid_t entryPointGlobalId = getGlobalDocId(group_base, doc_offset_base, entry_point);
    const void * entryPointVal = nullptr;
    if (likely(vector_retriever_.isValid() && !contain_feature_)) {
        if (!vector_retriever_(base_docid_ + entryPointGlobalId, entryPointVal)) {
            LOG_ERROR("retrieve vector failed. docid:%u", entryPointGlobalId);
            return -1;
        }
    } else {
        entryPointVal = feature_profile_->getInfo(entryPointGlobalId);
    }

    float dist = measure_(query_val, entryPointVal, len);
    compare_cnt++;

    for (int32_t level = max_level; level >= 1; --level) {
        if (updateEntryPoint(group_offset, group_base, doc_offset_base, query_val, len, level, entry_point, dist, compare_cnt, context) != 0) {
            return -1;
        }
    }

    if (searchNeighbors(group_offset, group_base, doc_offset_base, query_val, len, 0, entry_point, dist, topk_heap, compare_cnt, pCurGroup->doc_total_num, context, max_scan_num_in_query) != 0) {
        return -1;
    }

    return 0;
}

int CoarseHnswIndex::updateZeroEntryPoint(uint64_t group_offset, char* group_base, uint64_t *doc_offset_base, const void *query_val, size_t len, int32_t level, docid_t &entry_point,
                                      float &dist, uint64_t &compare_cnt)
{
    while (true) {
        //position base on neighborList, index begin with 1
        NeighborListHeader *header = getNeighborList(group_base, doc_offset_base, level, entry_point);
        int endPos = static_cast<int>(header->neighbor_cnt) - 1;
        if (endPos < 0) {
            return 0;
        }
        int beginPos = ((endPos + 1) >= step_) ? (endPos + 1  - step_) : 0;

        while (true) {
            bool findCloser = false;
            docid_t *curNeighbor = reinterpret_cast<docid_t *>(header + 1) + beginPos;
            for (int i = beginPos; i <= endPos; ++i) {
                docid_t node = *curNeighbor++;
                const void * nodePointVal = nullptr;
                if (likely(vector_retriever_.isValid())) {
                    if (!vector_retriever_(base_docid_ + node, nodePointVal)) {
                        LOG_ERROR("retrieve vector failed. docid:%u", node);
                        return -1;
                    }
                } else {
                    nodePointVal = feature_profile_->getInfo(node);
                }

                float curDist = measure_(query_val, nodePointVal, len);
                compare_cnt++;
                if (curDist < dist) {
                    entry_point = node;
                    dist = curDist;
                    findCloser = true;
                }
            }

            if (findCloser) {
                break;
            }
            //can't find any neighbor much closer to current entry point
            endPos = beginPos - 1;
            if (endPos < 0) {
                return 0;
            }
            beginPos = ((endPos + 1) >= step_) ? (endPos + 1 - step_) : 0;
        }
    }
    return 0;
}

int CoarseHnswIndex::searchZeroHnswNeighbors(uint64_t group_offset, TopkHeap &topk_heap, const void *query_val, size_t len, uint64_t &compare_cnt, int max_scan_num_in_query) {

    // 提前计算偏移，减少后续重复计算
    char* group_base = p_base_ + group_offset;
    uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(group_base + sizeof(GroupHeader));

    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);
    docid_t entry_point = pCurGroup->entry_point;
    int32_t max_level = static_cast<int32_t>(pCurGroup->cur_max_level);
    bool is_hnsw_group = pCurGroup->is_hnsw;
    if (!is_hnsw_group) {
        LOG_ERROR("Not a HNSW group!");
        return -1;
    }
    
    if (unlikely(entry_point == std::numeric_limits<docid_t>::max())) {
        LOG_ERROR("No entry point!");
        return -1;
    }

    const void * entryPointVal = nullptr;
    if (likely(vector_retriever_.isValid() && !contain_feature_)) {
        if (!vector_retriever_(base_docid_ + entry_point, entryPointVal)) {
            LOG_ERROR("retrieve vector failed. docid:%u", entry_point);
            return -1;
        }
    } else {
        entryPointVal = feature_profile_->getInfo(entry_point);
    }

    float dist = measure_(query_val, entryPointVal, len);
    compare_cnt++;

    for (int32_t level = max_level; level >= 1; --level) {
        if (updateZeroEntryPoint(group_offset, group_base, doc_offset_base, query_val, len, level, entry_point, dist, compare_cnt) != 0) {
            return -1;
        }
    }

    VisitList<uint32_t> visit_list;
    int max_scan_num = (max_scan_num_in_query ? max_scan_num_in_query : max_scan_num_);
    visit_list.init(pCurGroup->doc_total_num, max_scan_num);
    CandidateHeap candidates;
    candidates.reset(max_scan_num);
    if (searchZeroNeighbors(group_base, doc_offset_base, query_val, len, 0, entry_point, dist, topk_heap, compare_cnt, visit_list, candidates) != 0) {
        return -1;
    }

    return 0;
}

int CoarseHnswIndex::searchBruteNeighbors(uint64_t group_offset, std::vector<DistNode> &dist_nodes, const void *query_val, size_t len, uint64_t &compare_cnt) {
    
    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(p_base_ + group_offset);
    uint32_t doc_total_num = pCurGroup->doc_total_num;
    dist_nodes.resize(doc_total_num);
    bool is_hnsw_group = pCurGroup->is_hnsw;
    if (is_hnsw_group) {
        LOG_ERROR("This is a HNSW group!");
        return -1;
    }

    docid_t *doc_id_base = reinterpret_cast<docid_t *>(pCurGroup + 1);

    for (uint32_t i = 0; i < doc_total_num; i++) {
        docid_t docGlobalId = doc_id_base[i];
        const void *docVal = nullptr;
        if (likely(vector_retriever_.isValid() && !contain_feature_)) {
            if (!vector_retriever_(base_docid_ + docGlobalId, docVal)) {
                LOG_ERROR("retrieve vector failed. docid:%u", docGlobalId);
                return -1;
            }
        } else {
            docVal = feature_profile_->getInfo(docGlobalId);
        }
        float dist = measure_(query_val, docVal, len);
        dist_nodes.at(i).key = docGlobalId;
        dist_nodes.at(i).dist = dist;
    }
    compare_cnt += doc_total_num;
    return 0;
}

int CoarseHnswIndex::bruteSearchHnswNeighbors(uint64_t group_offset, std::vector<DistNode> &dist_nodes, const void *query_val, size_t len, uint64_t &compare_cnt) {

    // 提前计算偏移，减少后续重复计算
    char* group_base = p_base_ + group_offset;
    uint64_t *doc_offset_base = reinterpret_cast<uint64_t *>(group_base + sizeof(GroupHeader));

    GroupHeader *pCurGroup = reinterpret_cast<GroupHeader *>(group_base);

    uint32_t doc_total_num = pCurGroup->doc_total_num;
    dist_nodes.resize(doc_total_num);

    bool is_hnsw_group = pCurGroup->is_hnsw;
    if (!is_hnsw_group) {
        LOG_ERROR("Not a HNSW group!");
        return -1;
    }

    for (docid_t i = 0; i < doc_total_num; i++) {
        docid_t docGlobalId = getGlobalDocId(group_base, doc_offset_base, i);
        const void *docVal = nullptr;
        if (likely(vector_retriever_.isValid() && !contain_feature_)) {
            if (!vector_retriever_(base_docid_ + docGlobalId, docVal)) {
                LOG_ERROR("retrieve vector failed. docid:%u", docGlobalId);
                return -1;
            }
        } else {
            docVal = feature_profile_->getInfo(docGlobalId);
        }
        float dist = measure_(query_val, docVal, len);
        compare_cnt++;
        dist_nodes.at(i).key = docGlobalId;
        dist_nodes.at(i).dist = dist;
    }

    return 0;
}

MERCURY_NAMESPACE_END(core);