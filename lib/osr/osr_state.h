#ifndef OSR_STATE_H
#define OSR_STATE_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tuple>

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
typedef Eigen::Vector3d AngleAxisVector;
typedef Eigen::Matrix<double, -1, kStateDimension> ArrayOfStates; // State per-ROW
typedef Eigen::Matrix<double, -1, 3> ArrayOfTrans;
typedef Eigen::Matrix<double, -1, 3> ArrayOfAA;
typedef Eigen::Matrix4d StateMatrix;
using Transform = Eigen::Transform<double, 3, Eigen::AffineCompact>;

osr::Transform translate_state_to_transform(const StateVector& state);

/*
 * Interpolate between two SE3 states
 */
StateVector interpolate(const StateVector& pkey,
	                const StateVector& nkey,
	                StateScalar tau);
std::tuple<StateTrans, StateQuat> decompose(const StateVector&);
StateVector compose(const StateTrans&, const StateQuat&);
double distance(const StateVector& lhv, const StateVector& rhv);
Eigen::VectorXd multi_distance(const StateVector& origin, const ArrayOfStates& targets);
Eigen::Matrix3d extract_rotation_matrix(const StateVector&);

std::tuple<StateTrans, AngleAxisVector, StateVector>
differential(const StateVector& from, const StateVector& to);

StateVector
apply(const StateVector& from, const StateTrans& tr, const AngleAxisVector& aa);

StateTrans action_to_axis(int action);
Eigen::MatrixXf get_permutation_to_world(const Eigen::MatrixXf& views, int view);

}

#endif
