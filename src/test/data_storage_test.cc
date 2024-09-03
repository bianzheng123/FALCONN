#include "falconn/core/data_storage.h"

#include <vector>

#include "gtest/gtest.h"

namespace fc = falconn::core;

using falconn::DenseVector;
using fc::ArrayDataStorage;
using fc::PlainArrayDataStorage;
using fc::TransformedDataStorage;
using std::vector;


TEST(DataStorageTest, PlainArrayTest1) {
    typedef DenseVector<double> Vec;

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int num_points = 3;
    int dim = 2;

    PlainArrayDataStorage<Vec> ds(data, num_points, dim);

    ASSERT_EQ(ds.size(), num_points);

    auto iter = ds.get_full_sequence();

    ASSERT_TRUE(iter.is_valid());
    EXPECT_EQ(iter.get_point()[0], 1.0);
    EXPECT_EQ(iter.get_point()[1], 2.0);

    ++iter;
    ASSERT_TRUE(iter.is_valid());
    EXPECT_EQ(iter.get_point()[0], 3.0);
    EXPECT_EQ(iter.get_point()[1], 4.0);

    ++iter;
    ASSERT_TRUE(iter.is_valid());
    EXPECT_EQ(iter.get_point()[0], 5.0);
    EXPECT_EQ(iter.get_point()[1], 6.0);

    ++iter;
    ASSERT_FALSE(iter.is_valid());

    for (int ii = 0; ii < num_points; ++ii) {
        EXPECT_EQ(data[ii], ii + 1.0);
    }
}

TEST(DataStorageTest, PlainArrayTest2) {
    typedef DenseVector<double> Vec;

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int num_points = 3;
    int dim = 2;

    PlainArrayDataStorage<Vec> ds(data, num_points, dim);

    ASSERT_EQ(ds.size(), num_points);

    std::vector<int32_t> keys = {0, 2};
    auto iter = ds.get_subsequence(keys);

    ASSERT_TRUE(iter.is_valid());
    EXPECT_EQ(iter.get_point()[0], 1.0);
    EXPECT_EQ(iter.get_point()[1], 2.0);

    ++iter;
    ASSERT_TRUE(iter.is_valid());
    EXPECT_EQ(iter.get_point()[0], 5.0);
    EXPECT_EQ(iter.get_point()[1], 6.0);

    ++iter;
    ASSERT_FALSE(iter.is_valid());

    for (int ii = 0; ii < num_points; ++ii) {
        EXPECT_EQ(data[ii], ii + 1.0);
    }
}
