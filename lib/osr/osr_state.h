#ifndef OSR_STATE_H
#define OSR_STATE_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace osr {
constexpr int kActionDimension = 3;
/* *2 for pos/neg */
constexpr int kActionPerTransformType = kActionDimension * 2;
/* *2 for translation/rotation */
constexpr int kTotalNumberOfActions = kActionPerTransformType * 2;
/* 3 for Translation, 4 for quaternion in SO(3) */
constexpr int kStateDimension = 3 + 4;

typedef double StateScalar;
typedef Eigen::Vector3d StateTrans;
typedef Eigen::Quaternion<double> StateQuat;
typedef Eigen::Matrix<double, kStateDimension, 1> StateVector; // Column vector
typedef Eigen::Matrix<double, -1, kStateDimension> ArrayOfStates; // State per-ROW
typedef Eigen::Matrix4d StateMatrix;
using Transform = Eigen::Transform<double, 3, Eigen::AffineCompact>;

osr::Transform translate_state_to_transform(const StateVector& state);

/*
 * Interpolate between two SE3 states
 */
StateVector interpolate(const StateVector& pkey,
	                const StateVector& nkey,
	                StateScalar tau);
double distance(const StateVector& lhv, const StateVector& rhv);

}

#endif
