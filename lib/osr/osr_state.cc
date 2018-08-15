#include "osr_state.h"
#include <iostream>
#define GLM_FORCE_RADIANS
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

std::tuple<StateTrans, StateQuat>
decompose(const StateVector& state)
{
	StateQuat rot(state(3), state(4), state(5), state(6));
	StateTrans trans(state(0), state(1), state(2));
	rot.normalize();
	return std::make_tuple(trans, rot);
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

std::tuple<StateTrans, AngleAxisVector>
differential(const StateVector& from, const StateVector& to)
{
	StateTrans tr = to.segment<3>(0) - from.segment<3>(0);
	StateQuat rot_from(from(3), from(4), from(5), from(6));
	StateQuat rot_to(to(3), to(4), to(5), to(6));
	StateQuat rot_delta = rot_to * rot_from.inverse();
	Eigen::AngleAxis<StateScalar> aa(rot_delta);
	AngleAxisVector aav = aa.axis() * aa.angle();
	return std::make_tuple(tr, aav);
}

StateVector
apply(const StateVector& from, const StateTrans& tr, const AngleAxisVector& aa)
{
	auto tup = decompose(from);
	StateTrans trans = std::get<0>(tup);
	StateQuat rot = std::get<1>(tup);
	trans += tr;
	double angle = aa.norm();
	AngleAxisVector axis;
	if (angle == 0.0)
		axis << 1.0, 0.0, 0.0;
	else
		axis = aa.normalized();
	rot = StateQuat(Eigen::AngleAxis<StateScalar>(angle, axis)) * rot;
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


}
