#include "falconn/lsh_nn_table.h"

#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using falconn::compute_number_of_hash_functions;
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQuery;
using falconn::LSHNearestNeighborQueryPool;
using falconn::get_default_parameters;
using falconn::StorageHashTable;
using std::make_pair;
using std::unique_ptr;
using std::vector;

// Point dimension is 4
void basic_test_dense_1(const LSHConstructionParameters& params) {
  typedef DenseVector<float> Point;
  int dim = 4;

  Point p1(dim);
  p1[0] = 1.0;
  p1[1] = 0.0;
  p1[2] = 0.0;
  p1[3] = 0.0;
  Point p2(dim);
  p2[0] = 0.6;
  p2[1] = 0.8;
  p2[2] = 0.0;
  p2[3] = 0.0;
  Point p3(dim);
  p3[0] = 0.0;
  p3[1] = 0.0;
  p3[2] = 1.0;
  p3[3] = 0.0;
  vector<Point> points;
  points.push_back(p1);
  points.push_back(p2);
  points.push_back(p3);

  unique_ptr<LSHNearestNeighborTable<Point>> table(
      construct_table<Point>(points, params));
  unique_ptr<LSHNearestNeighborQuery<Point>> query(
      table->construct_query_object());

  int32_t res1 = query->find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  int32_t res2 = query->find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  int32_t res3 = query->find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  Point p4(dim);
  p4[0] = 0.0;
  p4[1] = 1.0;
  p4[2] = 0.0;
  p4[3] = 0.0;
  int32_t res4 = query->find_nearest_neighbor(p4);
  EXPECT_EQ(1, res4);

  unique_ptr<LSHNearestNeighborQueryPool<Point>> query_pool(
      table->construct_query_pool());

  // Same queries as above but now through a query pool
  res1 = query_pool->find_nearest_neighbor(p1);
  EXPECT_EQ(0, res1);
  res2 = query_pool->find_nearest_neighbor(p2);
  EXPECT_EQ(1, res2);
  res3 = query_pool->find_nearest_neighbor(p3);
  EXPECT_EQ(2, res3);

  res4 = query_pool->find_nearest_neighbor(p4);
  EXPECT_EQ(1, res4);
}

TEST(WrapperTest, DenseHPTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::EuclideanSquared;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, DenseCPTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::EuclideanSquared;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.k = 2;
  params.l = 8;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, FlatHashTableTest1) {
  int dim = 4;
  LSHConstructionParameters params;
  params.dimension = dim;
  params.lsh_family = LSHFamily::Hyperplane;
  params.distance_function = DistanceFunction::EuclideanSquared;
  params.storage_hash_table = StorageHashTable::FlatHashTable;
  params.k = 2;
  params.l = 4;
  params.num_setup_threads = 0;

  basic_test_dense_1(params);
}

TEST(WrapperTest, ComputeNumberOfHashFunctionsTest) {
  typedef DenseVector<float> VecDense;

  LSHConstructionParameters params;
  params.dimension = 10;
  params.lsh_family = LSHFamily::Hyperplane;

  compute_number_of_hash_functions<VecDense>(5, &params);
  EXPECT_EQ(5, params.k);

  params.lsh_family = LSHFamily::Hyperplane;
  compute_number_of_hash_functions<VecDense>(5, &params);
  EXPECT_EQ(1, params.k);
}

TEST(WrapperTest, GetDefaultParametersTest1) {
  typedef DenseVector<float> Vec;

  LSHConstructionParameters params = get_default_parameters<Vec>(
      1000000, 128, DistanceFunction::EuclideanSquared, true);

  EXPECT_EQ(10, params.l);
  EXPECT_EQ(128, params.dimension);
  EXPECT_EQ(DistanceFunction::EuclideanSquared, params.distance_function);
  EXPECT_EQ(LSHFamily::Hyperplane, params.lsh_family);
  EXPECT_EQ(3, params.k);
  EXPECT_EQ(StorageHashTable::FlatHashTable,
            params.storage_hash_table);
  EXPECT_EQ(0, params.num_setup_threads);
}
