#include "osr_state.h"
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/io.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>

namespace osr {

osr::Transform translate_state_to_transform(const StateVector& state)
{
	StateQuat rot(state(3), state(4), state(5), state(6));
	StateTrans trans(state(0), state(1), state(2));
	osr::Transform tf;
	tf.setIdentity();
	tf.rotate(rot);
	tf.pretranslate(trans);
	return tf;
}

StateVector
interpolate(const StateVector& pkey,
            const StateVector& nkey,
            StateScalar tau)
{
	StateTrans p0(pkey(0), pkey(1), pkey(2));
	StateTrans p1(nkey(0), nkey(1), nkey(2));
	StateTrans pinterp = p0 * (1-tau) + p1 * tau;
	StateQuat Qfrom { pkey(3), pkey(4), pkey(5), pkey(6) };
	StateQuat Qto { nkey(3), nkey(4), nkey(5), nkey(6) };
	Qfrom.normalize();
	Qto.normalize();
	StateQuat Qinterp = Qfrom.slerp(tau, Qto);
	StateVector ret;
	ret << pinterp(0), pinterp(1), pinterp(2),
	       Qinterp.w(), Qinterp.x(), Qinterp.y(), Qinterp.z();
	return ret;
}

Eigen::VectorXd
path_metrics(const ArrayOfStates& qs)
{
	Eigen::VectorXd metrics;
	metrics.resize(qs.rows(), 1);
	double dist = 0.0;
	for (size_t i = 0; i < qs.rows(); i++) {
		if (i > 0) {
			dist += distance(qs.row(i-1), qs.row(i));
		}
		metrics(i) = dist;
	}
	return metrics;
}

StateVector
path_interpolate(const ArrayOfStates& qs,
                 const Eigen::VectorXd& metrics,
                 double tau)
{
	int key = -1;
	for (int i = 0; i < qs.rows() - 1; i++) {
		if (metrics(i) <= tau && tau < metrics(i+1)) {
			key = i;
			break;
		}
	}
	if (key < 0) {
		return qs.row(0);
	}
	double d = metrics(key+1) - metrics(key);
	double t;
	if (d < 1e-6) 
		t = 0.0;
	else
		t = (tau - metrics(key)) / d;
	return interpolate(qs.row(key), qs.row(key+1), t);
}

std::tuple<StateTrans, StateQuat>
decompose(const StateVector& state)
{
	StateQuat rot(state(3), state(4), state(5), state(6));
	StateTrans trans(state(0), state(1), state(2));
	rot.normalize();
	return std::make_tuple(trans, rot);
}

std::tuple<StateTrans, Eigen::Matrix3d>
decompose_2(const StateVector& state)
{
	auto dec = decompose(state);
	return std::make_tuple(std::get<0>(dec), std::get<1>(dec).toRotationMatrix());
}

std::tuple<StateTrans, StateScalar, StateTrans>
decompose_3(const StateVector& state)
{
	auto tup = decompose(state);
	Eigen::AngleAxis<StateScalar> aa(std::get<1>(tup));
	return std::make_tuple(std::get<0>(tup), aa.angle(), aa.axis());
}

StateVector
compose(const StateTrans& base, const StateQuat& irot)
{
	StateVector ret;
	StateQuat rot(irot);
	rot.normalize();
	ret << base(0), base(1), base(2),
	       rot.w(), rot.x(),
	       rot.y(), rot.z();
	return ret;
}

StateVector
compose_from_angleaxis(const StateTrans& tr,
                       double angle,
                       const StateTrans& axis)
{
	Eigen::AngleAxis<StateScalar> aa(angle, axis);
	return compose(tr, StateQuat(aa));
}

double distance(const StateVector& lhv, const StateVector& rhv)
{
	auto tup0 = decompose(lhv);
	auto tup1 = decompose(rhv);
	double trdist = (std::get<0>(tup0) - std::get<0>(tup1)).norm();
	double rotdist = std::get<1>(tup0).angularDistance(std::get<1>(tup1));
	// double trdist = (lhv.segment<3>(0) - rhv.segment<3>(0)).norm();
	// double dot = lhv.segment<4>(3).dot(rhv.segment<4>(3));
	// dot = std::max(-1.0, std::min(1.0, dot)); // Theoretically we don't need to but...
	// double rotdist = std::abs(std::acos(dot)); // |theta/2|
#if 0
	if (std::isnan(rotdist)) {
		std::cerr.precision(20);
		std::cerr << "[NAN] l: " << lhv.segment<4>(3) << std::endl;
		std::cerr << "[NAN] r: " << rhv.segment<4>(3) << std::endl;
		std::cerr << "[NAN] nl: " << lhv.segment<4>(3).normalized() << std::endl;
		std::cerr << "[NAN] nr: " << rhv.segment<4>(3).normalized() << std::endl;
		std::cerr << "[NAN] lnorm: " << lhv.segment<4>(3).norm() << std::endl;
		std::cerr << "[NAN] rnorm: " << rhv.segment<4>(3).norm() << std::endl;
		std::cerr << "[NAN] dot: " << dot << std::endl;
		throw std::runtime_error(std::string("YOUR SANITY IS BLASTED ") + __func__);
	}
#endif
	return trdist + rotdist;
}

Eigen::VectorXd
multi_distance(const StateVector& origin, const ArrayOfStates& targets)
{
	Eigen::VectorXd ret;
	const auto N = targets.rows();
	ret.resize(N);
	for (int i = 0; i < N; i++) {
		ret(i) = distance(origin, targets.row(i));
	}
	return ret;
}

Eigen::Matrix3d
extract_rotation_matrix(const StateVector& state)
{
	StateQuat rot(state(3), state(4), state(5), state(6));
	return rot.toRotationMatrix();
}

std::tuple<StateTrans, AngleAxisVector, StateVector>
differential(const StateVector& from, const StateVector& to)
{
	StateTrans tr = to.segment<3>(0) - from.segment<3>(0);
	StateQuat rot_from(from(3), from(4), from(5), from(6));
	StateQuat rot_to(to(3), to(4), to(5), to(6));
	StateQuat rot_delta = rot_to * rot_from.inverse();
	Eigen::AngleAxis<StateScalar> aa(rot_delta);
	AngleAxisVector aav = aa.axis() * aa.angle();
#if 0
	std::cerr << "AXIS: " << aa.axis() << std::endl << "ANGLE: " << aa.angle() << std::endl;
	std::cerr << "ROT_DELTA " << rot_delta.w() << ' ' << rot_delta.x() << ' ' << rot_delta.y() << ' ' << rot_delta.z() << std::endl;
	if (!rot_to.isApprox(StateQuat(aa) * rot_from)) {
		std::cerr << "PANIC: aa <-> quat error\n";
		std::cerr << "\t EXPECT " <<  rot_to.w() << ' ' << rot_to.x() << ' ' << rot_to.y() << ' ' << rot_to.z() << std::endl;
		Eigen::AngleAxis<StateScalar> eaa(rot_to);
		rot_to.normalize();
		std::cerr << "\t EXPECT (N) " <<  rot_to.w() << ' ' << rot_to.x() << ' ' << rot_to.y() << ' ' << rot_to.z() << std::endl;

		std::cerr << "\t EXPECT (AA) " <<  eaa.axis() << '\t' << eaa.angle() << '\n';
		StateQuat tquat = StateQuat(aa) * rot_from;
		std::cerr << "\t GOT " << tquat.w() << ' ' << tquat.x() << ' ' << tquat.y() << ' ' << tquat.z() << std::endl;
		eaa = tquat;
		std::cerr << "\t GOT (AA) " <<  eaa.axis() << '\t' << eaa.angle() << '\n';
		tquat.normalize();
		std::cerr << "\t GOT (N)" << tquat.w() << ' ' << tquat.x() << ' ' << tquat.y() << ' ' << tquat.z() << std::endl;
	}
	if (!rot_to.isApprox(rot_delta * rot_from)) {
		std::cerr << "PANIC: rot_delta error\n";
	}
#endif
	if (rot_to.angularDistance(aa * rot_from) > 1e-3)
		throw std::runtime_error("differential is buggy");
	{
		AngleAxisVector axis = aav.normalized();
		double angle = aav.norm();
		StateQuat rec_to = StateQuat(Eigen::AngleAxis<StateScalar>(angle, axis)) * rot_from;
		if (rot_to.angularDistance(rec_to) > 1e-3)
			throw std::runtime_error("differential is buggy in aa->quat");
#if 0
		std::cout << "\t aav " << aav.transpose() << std::endl;
		std::cout << "\t rot_to " <<  rot_to.w() << ' ' << rot_to.x() << ' ' << rot_to.y() << ' ' << rot_to.z() << std::endl;
		std::cout << "\t rec_to " <<  rec_to.w() << ' ' << rec_to.x() << ' ' << rec_to.y() << ' ' << rec_to.z() << std::endl;
		std::cout << std::flush;
#endif
	}
	return std::make_tuple(tr, aav, compose(tr, rot_delta));
}

Eigen::Matrix<StateScalar, -1, -1>
multi_differential(const StateVector& from, const ArrayOfStates& tos, bool with_se3)
{
	Eigen::Matrix<StateScalar, -1, -1> ret;
	size_t ROW = tos.rows();
	size_t COL = kActionDimension * 2; // Trans + Rotation, assume dim(Rotation) == dim(Trans)
	if (with_se3)
		COL += kStateDimension;
	ret.resize(ROW, COL);
	for (size_t i = 0; i < ROW; i++) {
		auto tup = differential(from, tos.row(i).transpose());
		ret.block<1,3>(i, 0) = std::get<0>(tup).transpose();
		ret.block<1,3>(i, 3) = std::get<1>(tup).transpose();
		if (with_se3)
			ret.block<1,kStateDimension>(i, 6) = std::get<2>(tup).transpose();
	}
	return ret;
}

StateVector
apply(const StateVector& from, const StateTrans& tr, const AngleAxisVector& aa)
{
	// std::cout << "\t apply.aav " << aa.transpose() << std::endl;
	auto tup = decompose(from);
	StateTrans trans = std::get<0>(tup);
	StateQuat rot_from = std::get<1>(tup);
	trans += tr;
	double angle = aa.norm();
	AngleAxisVector axis;
	if (angle == 0.0)
		axis << 1.0, 0.0, 0.0;
	else
		axis = aa.normalized();
	StateQuat rot = StateQuat(Eigen::AngleAxis<StateScalar>(angle, axis)) * rot_from;
	return compose(trans, rot);
}

StateTrans action_to_axis(int action)
{
	StateTrans tfvec { StateTrans::Zero() };
	float sym = action % 2 == 0 ? 1.0f : -1.0f;
	int axis_id = (action % kActionPerTransformType) / 2;
	tfvec(axis_id) = sym;
	return tfvec;
}

Eigen::MatrixXf
get_permutation_to_world(const Eigen::MatrixXf& views, int view)
{
	Eigen::MatrixXf ret;
	ret.setIdentity(kTotalNumberOfActions, kTotalNumberOfActions);
	if (view >= kTotalNumberOfActions)
		return ret;
	/*
	 * World translation/rotation axes.
	 */
	glm::vec4 world_axes[kTotalNumberOfActions];
	glm::vec4 viewed_axes[kTotalNumberOfActions];
	for (int i = 0; i < kTotalNumberOfActions; i++) {
		auto axis = action_to_axis(i);
		world_axes[i] = glm::vec4(axis(0), axis(1), axis(2), 0.0f);
	}
	glm::mat4 camera_rot = glm::mat4(1.0f);
	camera_rot = glm::rotate(camera_rot,
	                         glm::radians(views(view, 0)),
	                         glm::vec3(1.0f, 0.0f, 0.0f));
	camera_rot = glm::rotate(camera_rot,
	                         glm::radians(views(view, 1)),
	                         glm::vec3(0.0f, 1.0f, 0.0f));
	const float eyeDist = 2.0f;
	glm::vec4 eye = camera_rot * glm::vec4(0.0f, 0.0f, eyeDist, 1.0f);
	glm::vec4 cen = camera_rot * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 upv = camera_rot * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	glm::mat4 vmat = glm::lookAt(
	                             glm::vec3(eye),     // eye
	                             glm::vec3(cen),     // CENter
	                             glm::vec3(upv)      // UP Vector
	                            );
	static_assert(kTotalNumberOfActions == 2 * kActionPerTransformType,
	              "Previous Assumption becomes invalid, change your code accordingly");
	for (int action_type = 0; action_type < 2; action_type++) {
		int abegin = kActionPerTransformType * action_type;
		int aend = abegin + kActionPerTransformType;
		for (int locala = abegin; locala < aend; locala++) {
			viewed_axes[locala] = vmat * world_axes[locala];
			int best_match = -1;
			float best_dot = -1.0f;
			for (int worlda = abegin; worlda < aend; worlda++) {
				float dot = glm::dot(world_axes[worlda], viewed_axes[locala]);
				if (std::abs(1.0f - dot) < std::abs(1.0f - best_dot)) {
					best_dot = dot;
					best_match = worlda;
				}
			}
			if (best_match < 0) {
				throw std::runtime_error("CAVEAT: Cannot match Action "
							 + std::to_string(locala));
			}
			if (best_dot < 0.9f) {
				throw std::runtime_error("CAVEAT: Failed to match Action "
							 + std::to_string(locala)
							 + " dot is too large: "
							 + std::to_string(best_dot));
			}
			ret(locala, locala) = 0.0f;
			ret(best_match, locala) = 1.0f;
		}
	}
	Eigen::VectorXf ones = Eigen::VectorXf::Constant(kTotalNumberOfActions, 1.0f);
	Eigen::VectorXf csum = ret.colwise().sum();
	Eigen::VectorXf rsum = ret.rowwise().sum();
	std::cerr << "Permutation matrix for view " << view << std::endl
	          << ret << std::endl;
#if 0
	std::cerr << "CSUM " << csum << std::endl
		  << "RSUM " << rsum << std::endl;
#endif
	if (!csum.isApprox(ones))
		throw std::runtime_error("Permutation Matrix San check failed");
	if (!rsum.isApprox(ones))
		throw std::runtime_error("Permutation Matrix San check failed");
	return ret;
}

void state_vector_set_identity(StateVector& q)
{
	q << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;
}

StateVector get_identity_state_vector()
{
	StateVector q;
	state_vector_set_identity(q);
	return q;
}

}
