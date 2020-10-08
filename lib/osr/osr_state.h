/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
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
typedef Eigen::Matrix<double, kActionDimension, 1> ScaleVector;
typedef Eigen::Vector3d AngleAxisVector;
typedef Eigen::Matrix<double, -1, kStateDimension> ArrayOfStates; // State per-ROW
typedef Eigen::Matrix<StateScalar, -1, kActionDimension> ArrayOfPoints;
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
Eigen::VectorXd path_metrics(const ArrayOfStates& qs);
StateVector path_interpolate(const ArrayOfStates& qs,
                             const Eigen::VectorXd& metrics,
                             double tau);
std::tuple<StateTrans, StateQuat> decompose(const StateVector&);
std::tuple<StateTrans, Eigen::Matrix3d> decompose_2(const StateVector&);
std::tuple<StateTrans, StateScalar, StateTrans> decompose_3(const StateVector&);
StateVector compose(const StateTrans&, const StateQuat&);
StateVector compose_from_angleaxis(const StateTrans&, StateScalar angle, const StateTrans& axis);
double distance(const StateVector& lhv, const StateVector& rhv);
Eigen::VectorXd multi_distance(const StateVector& origin, const ArrayOfStates& targets);
Eigen::Matrix3d extract_rotation_matrix(const StateVector&);

std::tuple<StateTrans, AngleAxisVector, StateVector>
differential(const StateVector& from, const StateVector& to);

Eigen::Matrix<StateScalar, -1, -1>
multi_differential(const StateVector& from, const ArrayOfStates& tos, bool with_se3 = false);

StateVector
apply(const StateVector& from, const StateTrans& tr, const AngleAxisVector& aa);

StateTrans action_to_axis(int action);
Eigen::MatrixXf get_permutation_to_world(const Eigen::MatrixXf& views, int view);

void state_vector_set_identity(StateVector&);
StateVector get_identity_state_vector();

}

#endif
