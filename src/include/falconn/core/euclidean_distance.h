#ifndef __EUCLIDEAN_DISTANCE_H__
#define __EUCLIDEAN_DISTANCE_H__

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace falconn {
namespace core {

// Computes *SQUARED* Euclidean distance between dense vectors.

// The Dense functions assume that the data points are stored as dense
// Eigen column vectors.

template <typename CoordinateType = float>
struct EuclideanDistanceDense {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;

  template <typename Derived1, typename Derived2>
  CoordinateType operator()(const Eigen::MatrixBase<Derived1>& p1,
                            const Eigen::MatrixBase<Derived2>& p2) {
    return (p1 - p2).squaredNorm();
  }
};

}  // namespace core
}  // namespace falconn

#endif
