#include "falconn/core/composite_hash_table.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "falconn/core/flat_hash_table.h"
#include "test_utils.h"

namespace fc = falconn::core;
namespace ft = falconn::test;

using fc::FlatHashTable;
using fc::StaticCompositeHashTable;
using ft::check_result;
using std::pair;
using std::vector;

typedef StaticCompositeHashTable<uint32_t, int32_t, FlatHashTable<uint32_t>>
        StaticCompositeFlatHashTable;

TEST(CompositeHashTableTest, RetrieveTest1) {
    const int num_tables = 3;
    const int table_size = 10;
    FlatHashTable<uint32_t>::Factory factory(table_size);
    StaticCompositeFlatHashTable table(num_tables, &factory);

    vector<uint32_t> entries = {1, 0, 1, 8, 5, 2, 5};

    table.add_entries_for_table(entries, 0);
    table.add_entries_for_table(entries, 1);
    table.add_entries_for_table(entries, 2);

    vector<uint32_t> keys1 = {3};
    vector<uint32_t> keys2 = {8};
    vector<uint32_t> keys3 = {1};
    vector<vector<uint32_t>> all_keys = {keys1, keys2, keys3};

    std::vector<int32_t> expected_result1 = {0, 2, 3};
    auto result1 = table.retrieve_bulk(all_keys);
    check_result(result1, expected_result1);

    keys1 = keys2 = keys3 = {5};
    all_keys = {keys1, keys2, keys3};
    std::vector<int32_t> expected_result2 = {4, 4, 4, 6, 6, 6};
    auto result2 = table.retrieve_bulk(all_keys);
    check_result(result2, expected_result2);
}

TEST(CompositeHashTableTest, RetrieveMultiProbeTest1) {
    const int num_tables = 3;
    const int table_size = 10;
    FlatHashTable<uint32_t>::Factory factory(table_size);
    StaticCompositeFlatHashTable table(num_tables, &factory);

    vector<uint32_t> entriesa = {1, 0, 1, 8, 5, 2, 5};
    vector<uint32_t> entriesb = {0, 1, 2, 3, 4, 5, 6};
    table.add_entries_for_table(entriesa, 0);
    table.add_entries_for_table(entriesa, 1);
    table.add_entries_for_table(entriesb, 2);

    vector<uint32_t> keys1 = {3, 8, 1};
    vector<uint32_t> keys2 = {};
    vector<uint32_t> keys3 = {};
    vector<vector<uint32_t>> all_keys = {keys1, keys2, keys3};
    std::vector<int32_t> expected_result1 = {0, 2, 3};
    auto result1 = table.retrieve_bulk(all_keys);
    check_result(result1, expected_result1);

    keys1 = {};
    keys2 = {0, 2};
    keys3 = {2};
    all_keys = {keys1, keys2, keys3};
    std::vector<int32_t> expected_result2 = {1, 2, 5};
    auto result2 = table.retrieve_bulk(all_keys);
    check_result(result2, expected_result2);
}
