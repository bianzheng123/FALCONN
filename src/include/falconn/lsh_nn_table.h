#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

#include "falconn_global.h"

///
/// The main namespace.
///
namespace falconn {

// An exception class for errors occuring in the wrapper classes. Errors from
// the internal classes will throw other errors that also derive from
// FalconError.
class LSHNearestNeighborTableError : public FalconnError {
 public:
  LSHNearestNeighborTableError(const char* msg) : FalconnError(msg) {}
};

///
/// A common interface for query objects that execute table lookups such as
/// nearest neighbor queries. A query object does not change the state of the
/// parent LSHNearestNeighborTable.
///
template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborQuery {
 public:
  ///
  /// Sets the number of probes used for each query.
  /// The default setting is l (number of tables), which effectively disables
  /// multiprobing (the probing sequence only contains a single candidate per
  /// table).
  ///
  virtual void set_num_probes(int_fast64_t num_probes) = 0;
  ///
  /// Returns the number of probes used for each query.
  ///
  virtual int_fast64_t get_num_probes() = 0;

  ///
  /// Sets the maximum number of candidates considered in each query.
  /// The constant kNoMaxNumCandidates indicates that all candidates retrieved
  /// in the probing sequence should be considered. This is the default and
  /// usually a good setting. A maximum number of candidates is mainly useful
  /// to give worst case running time guarantees for every query.
  ///
  virtual void set_max_num_candidates(int_fast64_t max_num_candidates) = 0;
  ///
  /// Returns the maximum number of candidates considered in each query.
  ///
  virtual int_fast64_t get_max_num_candidates() = 0;

  ///
  /// Finds the key of the closest candidate in the probing sequence for q.
  ///
  virtual KeyType find_nearest_neighbor(const PointType& q) = 0;

  ///
  /// Find the keys of the k closest candidates in the probing sequence for q.
  /// The keys are returned in order of increasing distance to q.
  ///
  virtual void find_k_nearest_neighbors(const PointType& q, int_fast64_t k,
                                        std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys corresponding to candidates in the probing sequence for q
  /// that have distance at most threshold.
  ///
  virtual void find_near_neighbors(
      const PointType& q,
      typename PointTypeTraits<PointType>::ScalarType threshold,
      std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys of all candidates in the probing sequence for q.
  /// Every candidate key occurs only once in the result. The
  /// candidates are returned in the order of their first occurrence in the
  /// probing sequence.
  ///
  virtual void get_unique_candidates(const PointType& q,
                                     std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys of all candidates in the probing sequence for q. If a
  /// candidate key is found in multiple tables, it will appear multiple times
  /// in the result. The candidates are returned in the order in which they
  /// appear in the probing sequence.
  ///
  virtual void get_candidates_with_duplicates(const PointType& q,
                                              std::vector<KeyType>* result) = 0;

  ///
  /// Resets the query statistics.
  ///
  virtual void reset_query_statistics() = 0;

  ///
  /// Returns the current query statistics.
  ///
  /// TODO: figure out the right semantics here: should the average distance
  /// time be averaged over all queries or only the near(est) neighbor queries?
  ///
  virtual QueryStatistics get_query_statistics() = 0;

  virtual ~LSHNearestNeighborQuery() {}
};

///
/// A common interface for query pools. Query pools offer mostly the
/// same interface as an individual query object. The difference is that a
/// query pool keeps an internal set of query objects to execute the queries
/// potentially in parallel. The query pool is thread safe. This enables using
/// the query pool in combination with thread pools or parallel map
/// implementations where allocating a per-thread object is inconvenient or
/// impossible.
///
template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborQueryPool {
 public:
  ///
  /// Sets the number of probes used for each query.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual void set_num_probes(int_fast64_t num_probes) = 0;
  ///
  /// Returns the number of probes used for each query.
  ///
  virtual int_fast64_t get_num_probes() = 0;

  ///
  /// Sets the maximum number of candidates considered in each query.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual void set_max_num_candidates(int_fast64_t max_num_candidates) = 0;
  ///
  /// Returns the maximum number of candidates considered in each query.
  ///
  virtual int_fast64_t get_max_num_candidates() = 0;

  ///
  /// Finds the key of the closest candidate in the probing sequence for q.
  ///
  virtual KeyType find_nearest_neighbor(const PointType& q) = 0;

  ///
  /// Find the keys of the k closest candidates in the probing sequence for q.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual void find_k_nearest_neighbors(const PointType& q, int_fast64_t k,
                                        std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys corresponding to candidates in the probing sequence for q
  /// that have distance at most threshold.
  ///
  virtual void find_near_neighbors(
      const PointType& q,
      typename PointTypeTraits<PointType>::ScalarType threshold,
      std::vector<KeyType>* result) = 0;

  ///
  /// Returns the keys of all candidates in the probing sequence for q.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual void get_unique_candidates(const PointType& q,
                                     std::vector<KeyType>* result) = 0;

  ///
  /// Returns the multiset of all candidate keys in the probing sequence for q.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual void get_candidates_with_duplicates(const PointType& q,
                                              std::vector<KeyType>* result) = 0;

  ///
  /// Resets the query statistics.
  ///
  virtual void reset_query_statistics() = 0;

  ///
  /// Returns the current query statistics.
  /// See the documentation for LSHNearestNeighborQuery.
  ///
  virtual QueryStatistics get_query_statistics() = 0;

  virtual ~LSHNearestNeighborQueryPool() {}
};

///
/// Common interface shared by all LSH table wrappers.
///
/// The template parameter PointType should be one of the two point types
/// introduced above (DenseVector and SparseVector), e.g., DenseVector<float>.
///
/// The KeyType template parameter is optional and the default int32_t is
/// sufficient for up to 10^9 points.
///
template <typename PointType, typename KeyType = int32_t>
class LSHNearestNeighborTable {
 public:
  virtual void add_table() = 0;

  ///
  /// A special constant for set_max_num_candidates which is effectively
  /// equivalent to infinity.
  ///
  static const int_fast64_t kNoMaxNumCandidates = -1;

  ///
  /// This function constructs a new query object. The query object holds
  /// all per-query state and executes table lookups.
  ///
  /// num_probes == -1 (the default value) indicates that the number of probes
  /// should equal the number of tables. This corresponds to no multiprobe.
  ///
  /// max_num_candidates == -1 (the default value) indicates that the number of
  /// candidates should not be limited. This means that the entire probing
  /// sequence is used.
  ///
  virtual std::unique_ptr<LSHNearestNeighborQuery<PointType, KeyType>>
  construct_query_object(int_fast64_t num_probes = -1,
                         int_fast64_t max_num_candidates = -1) const = 0;

  ///
  /// This function constructs a new query pool. The query pool holds
  /// a set of query objects and supports an interface that can be safely
  /// called from multiple threads.
  ///
  /// num_probes == -1 (the default value) indicates that the number of probes
  /// should equal the number of tables. This corresponds to no multiprobe.
  ///
  /// max_num_candidates == -1 (the default value) indicates that the number of
  /// candidates should not be limited. This means that the entire probing
  /// sequence is used.
  ///
  /// num_query_objects <= 0 (the default value) indicates that the number of
  /// query objects should be 2 times the number of hardware threads (as
  /// indicated by std::thread:hardware_concurrency()). This is a reasonable
  /// default for thread pools etc. that do not use an excessive number of
  /// threads.
  ///
  virtual std::unique_ptr<LSHNearestNeighborQueryPool<PointType, KeyType>>
  construct_query_pool(int_fast64_t num_probes = -1,
                       int_fast64_t max_num_candidates = -1,
                       int_fast64_t num_query_objects = 0) const = 0;

  ///
  /// Virtual destructor.
  ///
  virtual ~LSHNearestNeighborTable() {}
};

///
/// Supported LSH families.
///
enum class LSHFamily {
  Unknown = 0,

  ///
  /// The hyperplane hash proposed in
  ///
  /// "Similarity estimation techniques from rounding algorithms"
  /// Moses S. Charikar
  /// STOC 2002
  ///
  Hyperplane = 1,

  ///
  /// The cross polytope hash first proposed in
  ///
  /// "Spherical LSH for Approximate Nearest Neighbor Search on Unit
  //   Hypersphere",
  /// Kengo Terasawa, Yuzuru Tanaka
  /// WADS 2007
  ///
  /// Our implementation uses the algorithmic improvements described in
  ///
  /// "Practical and Optimal LSH for Angular Distance",
  /// Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig
  ///   Schmidt
  /// NIPS 2015
  ///
  CrossPolytope = 2
};

///
/// Supported distance functions.
///
/// Note that we use distance functions only to filter the candidates in
/// find_nearest_neighbor, find_k_nearest_neighbors, and find_near_neighbors.
//  For only returning all the candidates, the distance function is irrelevant.
///
enum class DistanceFunction {
  Unknown = 0,

  ///
  /// The distance between p and q is -<p, q>. For unit vectors p and q,
  /// this means that the nearest neighbor to q has the smallest angle with q.
  ///
  NegativeInnerProduct = 1,

  ///
  /// The distance is the **squared** Euclidean distance (same order as the
  /// actual Euclidean distance / l2-distance).
  ///
  EuclideanSquared = 2
};

///
/// Supported low-level storage hash tables.
///
enum class StorageHashTable {
  Unknown = 0,
  ///
  /// The set of points whose hash is equal to a given one is retrieved using
  /// "naive" buckets. One table takes space O(#points + #bins).
  ///
  FlatHashTable = 1
};

///
/// Contains the parameters for constructing a LSH table wrapper. Not all fields
/// are necessary for all types of LSH tables.
///
struct LSHConstructionParameters {
  // Required parameters
  ///
  /// Dimension of the points. Required for all the hash families.
  ///
  int_fast32_t dimension = -1;
  ///
  /// Hash family. Required for all the hash families.
  ///
  LSHFamily lsh_family = LSHFamily::Unknown;
  ///
  /// Distance function. Required for all the hash families.
  ///
  DistanceFunction distance_function = DistanceFunction::Unknown;
  ///
  /// Number of hash function per table. Required for all the hash families.
  ///
  int_fast32_t k = -1;
  ///
  /// Number of hash tables. Required for all the hash families.
  ///
  int_fast32_t l = -1;
  ///
  /// Low-level storage hash table.
  ///
  StorageHashTable storage_hash_table = StorageHashTable::Unknown;
  ///
  /// Number of threads used to set up the hash table.
  /// Zero indicates that FALCONN should use the maximum number of available
  /// hardware threads (or 1 if this number cannot be determined).
  /// The number of threads used is always at most the number of tables l.
  ///
  int_fast32_t num_setup_threads = -1;
  ///
  /// Randomness seed.
  ///
  uint64_t seed = 409556018;

  // Optional parameters
  ///
  /// Dimension of the last of the k cross-polytopes. Required
  /// only for the cross-polytope hash.
  ///
  int_fast32_t last_cp_dimension = -1;
  ///
  /// Number of pseudo-random rotations. Required only for the
  /// cross-polytope hash.
  ///
  /// For sparse data, it is recommended to set num_rotations to 2.
  /// For sufficiently dense data, 1 rotation usually suffices.
  ///
  int_fast32_t num_rotations = -1;
  ///
  /// Intermediate dimension for feature hashing of sparse data. Ignored for
  /// the hyperplane hash. A smaller feature hashing dimension leads to faster
  /// hash computations, but the quality of the hash also degrades.
  /// The value -1 indicates that no feature hashing is performed.
  ///
  int_fast32_t feature_hashing_dimension = -1;
};

///
/// Computes the number of hash functions in order to get a hash with the given
/// number of relevant bits. For the cross polytope hash, the last cross
/// polytope dimension will also be determined. The input struct params must
/// contain valid values for the following fields:
///   - lsh_family
///   - dimension (for the cross polytope hash)
///   - feature_hashing_dimension (for the cross polytope hash with sparse
///     vectors)
/// The function will then set the following fields of params:
///   - k
///   - last_cp_dim (for the cross polytope hash, both dense and sparse)
///
template <typename PointType>
void compute_number_of_hash_functions(int_fast32_t number_of_hash_bits,
                                      LSHConstructionParameters* params);

///
/// A function that sets default parameters based on the following
/// dataset properties:
///
/// - the size of the dataset (i.e., the number of points)
/// - the dimension
/// - the distance function
/// - and a flag indicating whether the dataset is sufficiently dense
///   (for dense data, fewer pseudo-random rotations suffice)
///
/// The parameters should be reasonable for _sufficiently nice_ datasets.
/// If the dataset has special structure, or you want to maximize the
/// performance of FALCONN, you need to set the parameters by hand.
/// See the documentation and the GloVe example to make sense of the parameters.
///
/// In particular, the default setting should give very fast preprocessing
/// time. If you are willing to spend more time building the data structure to
/// improve the query time, you should increase l (the number of tables) after
/// calling this function.
///
template <typename PointType>
LSHConstructionParameters get_default_parameters(
    int_fast64_t dataset_size, int_fast32_t dimension,
    DistanceFunction distance_function, bool is_sufficiently_dense);

///
/// An exception class for errors occuring while setting up the LSH table
/// wrapper.
///
class LSHNNTableSetupError : public FalconnError {
 public:
  LSHNNTableSetupError(const char* msg) : FalconnError(msg) {}
};

///
/// Function for constructing an LSH table wrapper. The template parameters
/// PointType and KeyType are as in LSHNearestNeighborTable above. The
/// PointSet template parameter default is set so that a std::vector<PointType>
/// can be passed as the set of points for which a LSH table should be
/// constructed.
///
/// For dense data stored in a single large array, you can also use the
/// PlainArrayPointSet struct as the PointSet template parameter in order to
/// pass a densly stored data array.
///
/// The points object *must* stay valid for the lifetime of the LSH table.
///
/// The caller assumes ownership of the returned pointer.
///
template <typename PointType, typename KeyType = int32_t,
          typename PointSet = std::vector<PointType>>
std::unique_ptr<LSHNearestNeighborTable<PointType, KeyType>> construct_table(
    const PointSet& points, const LSHConstructionParameters& params);

}  // namespace falconn

#include "wrapper/cpp_wrapper_impl.h"

#endif
