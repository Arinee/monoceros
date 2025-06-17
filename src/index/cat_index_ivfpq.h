#pragma once

#include "index_ivfpq.h"
#include "query_distance_matrix.h"
#include "general_search_context.h"

namespace mercury {

class CatIndexIvfpq : public IndexIvfpq
{
public:
    typedef std::shared_ptr<CatIndexIvfpq> Pointer;

    CatIndexIvfpq() {
        _keyCatMap = std::make_shared<HashTable<key_t, cat_t, 1>>();
        _catSet = std::make_shared<HashTable<cat_t, cat_t>>();
    }

    std::shared_ptr<HashTable<key_t, cat_t, 1>> GetKeyCatMap() const { return _keyCatMap; }
    std::shared_ptr<HashTable<cat_t, cat_t>> GetCatSet() const { return _catSet; }

    bool Load(IndexStorage::Handler::Pointer &&file_handle) override;

    GENERATE_RETURN_EMPTY_INDEX(CatIndexIvfpq);

protected:
    std::shared_ptr<HashTable<key_t, cat_t, 1>> _keyCatMap;
    std::shared_ptr<HashTable<cat_t, cat_t>> _catSet;
};
} // namespace mercury
