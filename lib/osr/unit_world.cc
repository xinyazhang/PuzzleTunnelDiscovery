#include "unit_world.h"
#include "scene.h"
#include "cdmodel.h"
#include <iostream>
#include <glm/gtx/io.hpp>
#include <atomic>
#include <stdexcept>
#include <random>
#include <igl/doublearea.h>
#include <igl/cross.h>
#include <igl/barycentric_coordinates.h>
#include <igl/writeOBJ.h>
#include <tritri/tritri_igl.h>
#include <tritri/tritri_cop.h>
#if PYOSR_HAS_CGAL
#include <meshbool/join.h>
#endif

#include "ode_data.h"

namespace osr {

using std::tie;

const uint32_t UnitWorld::GEO_ENV;
const uint32_t UnitWorld::GEO_ROB;

auto glm2Eigen(const glm::mat4& m)
{
	StateMatrix ret;
	ret.setZero();
	// GLM uses column major
	ret << m[0][0], m[1][0], m[2][0], m[3][0],
	       m[0][1], m[1][1], m[2][1], m[3][1],
	       m[0][2], m[1][2], m[2][2], m[3][2],
	       m[0][3], m[1][3], m[2][3], m[3][3];
	return ret;
}

auto glm2Eigen(const glm::vec3& v)
{
	Eigen::Vector3d rv;
	rv << v[0], v[1], v[2];
	return rv;
}


UnitWorld::UnitWorld()
{
	perturbate_.setZero();
	perturbate_(3) = 1.0;
	perturbate_tf_.setIdentity();
}

UnitWorld::~UnitWorld()
{
}

void
UnitWorld::copyFrom(const UnitWorld* other)
{
	shared_ = true;
	scene_.reset(new Scene(other->scene_));
	cd_scene_.reset(new CDModel(*scene_));
	if (other->robot_) {
		robot_.reset(new Scene(other->robot_));
		cd_robot_.reset(new CDModel(*robot_));
	} else {
		robot_.reset();
	}
	scene_scale_ = other->scene_scale_;
	calib_mat_ = glm2Eigen(scene_->getCalibrationTransform());
	inv_calib_mat_ = calib_mat_.inverse();
	perturbate_ = other->perturbate_;
}

void
UnitWorld::loadModelFromFile(const std::string& fn)
{
	scene_.reset(new Scene);
	const glm::vec3 blue(0.0f, 0.0f, 1.0f);
	scene_->load(fn, &blue);
}

void
UnitWorld::loadRobotFromFile(const std::string& fn)
{
	robot_.reset(new Scene);
	const glm::vec3 red(1.0f, 0.0f, 0.0f);
	robot_->load(fn, &red);
	robot_state_.setZero();
	robot_state_(3) = 1.0; // Quaternion for no rotation
}


void
UnitWorld::enforceRobotCenter(const StateTrans& center)
{
	robot_->overrideCenter({center(0), center(1), center(2)});
}


/*
 * scaleToUnit: calculate the scaling factor
 */
void
UnitWorld::scaleToUnit()
{
	auto scene_span =  scene_->getBoundingBox().span();
	float robot_span = 1.0f;
	if (robot_)
		robot_span =  robot_->getBoundingBox().span();
	scene_scale_ = 1.0 / std::max(scene_span, robot_span);
}


/*
 * angleModel: acutally calculate the transformation to unit world
 */
void
UnitWorld::angleModel(float latitude, float longitude)
{
	if (latitude != 0.0 or longitude != 0.0) {
		throw std::runtime_error("The arguments of UnitWorld::angleModel are resevered for compatibility reason. Both values must be 0.");
	}
	scene_->resetTransform();
	// std::cerr << "Before scale " << scene_->getCalibrationTransform() << std::endl;
	scene_->scale(glm::vec3(scene_scale_));
	// std::cerr << "After scale " << scene_->getCalibrationTransform() << std::endl;
	scene_->rotate(glm::radians(latitude), 1, 0, 0);      // latitude
	scene_->rotate(glm::radians(longitude), 0, 1, 0);     // longitude
	calib_mat_ = glm2Eigen(scene_->getCalibrationTransform());
	inv_calib_mat_ = calib_mat_.inverse();
	cd_scene_.reset(new CDModel(*scene_));
	if (robot_) {
		robot_->resetTransform();
		robot_->moveToCenter();
		robot_->scale(glm::vec3(scene_scale_));
		robot_->rotate(glm::radians(latitude), 1, 0, 0);      // latitude
		robot_->rotate(glm::radians(longitude), 0, 1, 0);     // longitude
		cd_robot_.reset(new CDModel(*robot_));
	}
}


void
UnitWorld::setPerturbation(const StateVector& pert)
{
	perturbate_ = pert;
	perturbate_tf_ = translate_state_to_transform(pert);
}

StateVector
UnitWorld::getPerturbation() const
{
	return perturbate_;
}

void
UnitWorld::setRobotState(const StateVector& state)
{
	robot_state_ = state;
}

StateVector
UnitWorld::getRobotState() const
{
	return robot_state_;
}


StateTrans
UnitWorld::getModelCenter() const
{
	return glm2Eigen(scene_->getCenter());
}


std::tuple<Transform, Transform>
UnitWorld::getCDTransforms(const StateVector& state) const
{
	Transform envTf;
	Transform robTf;
#if 0
	envTf = getSceneMatrix();
	robTf = translate_state_to_transform(state);
	robTf = robTf * getRobotMatrix();
#else
	// envTf.setIdentity();
	envTf = translate_state_to_transform(perturbate_); // state already contains perturbation.
	robTf = translate_state_to_transform(state);
#endif
	return std::make_tuple(envTf, robTf);
}


bool
UnitWorld::isValid(const StateVector& state) const
{
	if (!cd_scene_ || !cd_robot_)
		return true;
	Transform envTf;
	Transform robTf;
	std::tie(envTf, robTf) = getCDTransforms(state);
	// std::cerr << "CD with\n" << envTf.matrix() << "\n\n" << robTf.matrix() << "\n";
	// std::cerr << translate_state_to_matrix(state) << '\n';
	// std::cerr << robTf.matrix() << '\n';
#if 0
	Eigen::Matrix4f robTfm, envTfm;
	auto tf = robot_->transform();
	auto etf = scene_->transform();
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			robTfm(i, j) = tf[i][j];
			envTfm(i, j) = etf[i][j];
		}
	}
	robTf = robTfm.block<3,4>(0,0);
	envTf = envTfm.block<3,4>(0,0);
	std::cerr << " Rob TF (glm):\n" << tf << "\n";
	std::cerr << " Rob TFm :\n" << robTfm << "\n";
	std::cerr << " Rob TF (before):\n" << robTf.matrix() << "\n";
#endif
#if 0
	std::cerr << " Env TF:\n" << envTf.matrix() << "\n";
	std::cerr << " Rob TF:\n" << robTf.matrix() << "\n";
#endif
	return !CDModel::collide(*cd_scene_, envTf, *cd_robot_, robTf);
}

bool UnitWorld::isDisentangled(const StateVector& state) const
{
	if (!cd_scene_ || !cd_robot_) {
		throw std::runtime_error("Pain in the ass: models not loaded");
		return true;
	}
	Transform envTf;
	Transform robTf;
#if 1
	std::tie(envTf, robTf) = getCDTransforms(state);
#else
	envTf.setIdentity();
	robTf.setIdentity();
#endif
	return !CDModel::collideBB(*cd_scene_, envTf, *cd_robot_, robTf);
}


std::tuple<StateVector, bool, float>
UnitWorld::transitState(const StateVector& state,
                       int action,
                       double transit_magnitude,
                       double verify_delta) const
{
	Eigen::Vector2f magnitudes;
	magnitudes << transit_magnitude , transit_magnitude * 2;
	Eigen::Vector2f deltas;
	deltas << verify_delta, verify_delta * 2;

	int action_type = action < kActionPerTransformType ? 0 : 1;
	if (action >= kTotalNumberOfActions) {
		/*
		 * Invalid action, return the initial state.
		 */
		return std::make_tuple(state, false, 0.0f);
	}
	if (!isValid(state)) {
		/*
		 * Invalid initial state
		 */
		std::cerr << "> Invalid Init State " << std::endl;
		return std::make_tuple(state, false, 0.0f);
	}
	auto magnitude = magnitudes(action_type);
	auto deltacap = deltas(action_type);
	float sym = action % 2 == 0 ? 1.0f : -1.0f;
	/*
	 * action (0-11) to XYZ (0-2)
	 * X: 0,1 or 6,7
	 * Y: 2,3 or 8,9
	 * Z: 4,5 or 10,11
	 */
	int axis_id = (action - kActionPerTransformType * action_type) / 2;
	StateQuat deltarot;
	StateTrans tfvec { StateTrans::Zero() };
	tfvec(axis_id) = 1.0f;
#if 1
	StateTrans trans;
	StateQuat rot;
	tie(trans, rot) = decompose(state);
	if (action_type == 0) {
		trans += deltacap * tfvec * sym;
	} else {
		Eigen::AngleAxis<StateScalar> aa(deltacap * sym, tfvec);
		rot = aa * rot;
	}
	StateVector to_state;
	to_state = compose(trans, rot);

	float current_verify_delta = verify_delta;
	// std::cerr << "> Init V Mag " << verify_delta << std::endl;
	while (true) {
		StateVector free_state;
		bool done;
		float prog;
		tie(free_state, done, prog) = transitStateTo(state, to_state, current_verify_delta);
		if (!isValid(free_state))
			throw std::runtime_error("SAN check failed, invalid state from transitStateTo");
		if (prog > 0.0 or current_verify_delta == 0.0)
			/* Sometimes vmag vanishes */
			return std::make_tuple(free_state, done, prog);
		/*
		 * A free state must be locally free, prog == 0.0 implies
		 * current verify delta is too large.
		 */
		current_verify_delta /= 2.0;
		// std::cerr << "> V Mag shrink to " << current_verify_delta << std::endl;
	}
#else
	/*
	 * Post condition:
	 *      tfvec or aa presents the delta action
	 */
	std::function<void(float)> applier;
	/* rot and trans are accumulators that would be initialized later */
	StateQuat rot;
	StateTrans trans;
	if (action_type == 0) {
		applier = [&rot, &trans, &tfvec](float delta) {
			trans += delta * tfvec;
		};
	} else {
		applier = [&rot, &trans, &tfvec, sym](float delta) {
			Eigen::AngleAxis<StateScalar> aa(delta * sym, tfvec);
			rot = aa * rot;
			rot.normalize();
		};
	}
	while (true) {
		StateVector nstate, freestate(state);
		bool done = true;
		auto accum = 0.0f;
		tie(trans, rot) = decompose(state);
		while (true) {
			float delta = std::min(deltacap, magnitude - accum);
			float naccum = accum + delta;
			applier(delta);
			nstate = compose(trans, rot);
			/*
			 * Verify the new state at accum + delta
			 */
			if (!isValid(nstate)) {
				done = false;
				break;
			}
			freestate = nstate;
			/*
			 * naccum is valid, proceed.
			 */
			accum = naccum;
			/* Exit when reaches last iteration */
			if (accum == magnitude)
				break;
		}
		if (accum > 0.0)
			break;
		/*
		 * Otherwise we need to reduce deltacap
		 */
	}
	float prog = accum / magnitude;
	return std::make_tuple(freestate, done, prog);
#endif
}


std::tuple<StateVector, bool, float>
UnitWorld::transitStateTo(const StateVector& from,
                          const StateVector& to,
                          double verify_delta) const
{
	auto tup = transitStateToWithContact(from, to, verify_delta);
	return std::make_tuple(std::get<0>(tup), std::get<2>(tup), std::get<3>(tup));
}


std::tuple<StateVector, StateVector, bool, float, float>
UnitWorld::transitStateToWithContact(const StateVector& from,
                                     const StateVector& to,
                                     double verify_delta) const
{
	double dist = distance(from, to);
	// std::cerr << "\t\tNSeg: " << nseg << std::endl;
#if 0 // Parallel Version
	int nseg = int(std::ceil(std::max(1.0, dist/verify_delta)));
	double rate = 1.0 / double(nseg);
	Eigen::VectorXi valid;
	valid.setZero(nseg);
	std::atomic<bool> hitInvalid(false);
	{
// #pragma omp parallel for
		for (int i = 1; i <= nseg; i++) {
			if (hitInvalid.load())
				continue;
			double tau = i * rate;
			auto state = interpolate(from, to, tau);
			if (!isValid(state)) {
				valid(i - 1) = 0;
				hitInvalid.store(true);
			} else {
				valid(i - 1) = 1;
			}
		}
	}
	for (int i = 0; i < nseg; i++) {
		if (!valid(i)) {
			/*
			 * Note: valid(0) records the validity of (i+1) * rate
			 * So i * rate is the correct tau
			 */
			double tau = i * rate;
			auto last_free = interpolate(from, to, tau);
			return std::make_tuple(last_free, false, tau);
		}
	}
#else
	if (verify_delta >= dist) {
		return std::make_tuple(from, to, false, 0.0, 1.0);
	}
	StateVector last_free = from;
	double delta = verify_delta;
	double inv_dist = 1.0/dist;
	double last_tau = 0.0;
	double tau = 0.0;
	StateVector state = from;
	while (delta <= dist) {
		double tau = delta * inv_dist;
		state = interpolate(from, to, tau);
		if (!isValid(state)) {
			return std::make_tuple(last_free, state, false, last_tau, tau);
		}
		last_tau = tau;
		last_free = state;
		if (delta < dist) {
			// make sure tau = 1.0 is checked.
			delta += verify_delta;
			delta = std::min(dist, delta);
		} else {
			// make sure tau = 1.0 is NOT double-checked.
			delta += verify_delta;
		}
	}
#endif
	// We may assert (delta > dist) == true
	return std::make_tuple(last_free, state, delta > dist, last_tau, tau);
}


bool
UnitWorld::isValidTransition(const StateVector& from,
                             const StateVector& to,
                             double initial_verify_delta) const
{
	double d = distance(from, to);
	auto vdelta = std::min(d / 2.0, initial_verify_delta);
	auto tup = transitStateTo(from, to, vdelta);
	return std::get<1>(tup);
}


std::tuple<StateVector, bool, float>
UnitWorld::transitStateBy(const StateVector& from,
	                  const StateTrans& tr,
	                  const AngleAxisVector& aav,
	                  double verify_delta) const
{
	StateTrans base;
	base << from(0), from(1), from(2);
	base += tr;
	StateQuat rot(from(3), from(4), from(5), from(6));
	Eigen::AngleAxis<StateScalar> aa(aav.norm(), aav.normalized());
	rot = aa * rot;
	StateVector to;
	to << base(0), base(1), base(2),
	      rot.w(), rot.x(),
	      rot.y(), rot.z();
	return transitStateTo(from, to ,verify_delta);
}


StateMatrix
UnitWorld::getSceneMatrix() const
{
	if (!scene_)
		return StateMatrix::Identity();
	return glm2Eigen(scene_->getCalibrationTransform());
}

StateMatrix
UnitWorld::getRobotMatrix() const
{
	if (!robot_)
		return StateMatrix::Identity();
	return glm2Eigen(robot_->getCalibrationTransform());
}


StateVector
UnitWorld::translateToUnitState(const StateVector& state) const
{
#if 1
	Eigen::Vector4d t(state(0), state(1), state(2), 1.0f);
	Eigen::Vector4d nt = calib_mat_ * t;
	StateVector ret;
	ret << nt(0), nt(1), nt(2),
               state(3), state(4), state(5), state(6);
	return applyPertubation(ret);
#else
	return state;
#endif
}

StateVector
UnitWorld::translateFromUnitState(const StateVector& pstate) const
{
#if 1
	auto state = unapplyPertubation(pstate);
	Eigen::Vector4d t(state(0), state(1), state(2), 1.0f);
	Eigen::Vector4d nt = inv_calib_mat_ * t;
	StateVector ret;
	ret << nt(0), nt(1), nt(2),
               state(3), state(4), state(5), state(6);
	return ret;
#else
	return state;
#endif
}

ArrayOfStates
UnitWorld::translateUnitStateToOMPLState(const ArrayOfStates& qs, bool to_angle_axis) const
{
	int N = qs.rows();
	ArrayOfStates ret;
	ret.resize(qs.rows(), qs.cols());
	// Handle the case of using enforceRobotCenter
	Eigen::Vector3d delta_center = glm2Eigen(robot_->getOMPLCenter() - robot_->getCenter());
	delta_center *= scene_scale_;
	for (int i = 0; i < N; i++) {
		auto tup = decompose(qs.row(i).transpose());
		Eigen::Vector3d t_prime = std::get<0>(tup) + std::get<1>(tup) * delta_center;
		ret.row(i) = compose(t_prime, std::get<1>(tup));
		ret.row(i) = translateFromUnitState(ret.row(i));
	}
	// OMPL uses W last while we uses W first
	Eigen::Matrix<double, -1, 4> wfirst(N, 4);
	wfirst = ret.block(0, 3, N, 4);
	if (to_angle_axis) {
		// W first -> Angle Axis
		for (int i = 0; i < N; i++) {
			StateQuat rot(wfirst(i, 0), wfirst(i, 1), wfirst(i, 2), wfirst(i, 3));
			Eigen::AngleAxis<StateScalar> aa(rot);
			ret(i, 3 + 0) = aa.angle();
			ret.block<1, 3>(i, 3 + 1) = aa.axis().transpose();
		}
	} else {
		// W fist -> W last
		ret.col(3 + 3) = wfirst.col(0);
		ret.col(3 + 0) = wfirst.col(1);
		ret.col(3 + 1) = wfirst.col(2);
		ret.col(3 + 2) = wfirst.col(3);
	}
	return ret;
}

/*
 * Let R_{ompl} and t_{ompl} be the rotation matrix and translation vector for OMPL state.
 * R_{vanilla} and t_{vanilla} be the rotation matrix and translation vector for vanilla state
 *
 * We solve the following equation to get R_{ompl} and t_{ompl}
 *    R_{vanilla}v + t_{vanilla} == R_{ompl}(v - O) + t_{omlp} holds any vertex v \in R^3.
 * In which O is the OMPL center.
 *
 * Let v be 0, we have t_{vanilla} == t_{ompl} - R_{ompl} O.
 * Substitude the above to the original equation we have
 * R_{ompl} = R_{vanilla}
 * t_(ompl} = t_{vanilla} + R_{ompl} O
 */
ArrayOfStates
UnitWorld::translateVanillaStateToOMPLState(const ArrayOfStates& qs) const
{
	static_assert(kStateDimension == 7, "Only support SE(3) for now");
	int N = qs.rows();
	ArrayOfStates ret;
	ret.resize(qs.rows(), qs.cols());
	// R_{ompl} = R_{vanilla}
	ret.block(0, 3, N, 4) = qs.block(0, 3, N, 4);
	// Get OMPL center
	Eigen::Vector3d ompl_center = glm2Eigen(robot_->getOMPLCenter());
	for (int i = 0; i < N; i++) {
		// Vanilla state also uses w-last quaternion
		// TODO: Generalize to multi-body
		StateQuat rot(qs(i,6), qs(i,3), qs(i,4), qs(i,5));
		// Eigen::Vector3d roo = rot.normalized().toRotationMatrix() * ompl_center;
		Eigen::Vector3d roo = rot.normalized()._transformVector(ompl_center);
		// ret.block(i, 0, 1, 3) = roo; <- this causes segfault
		ret(i, 0) = qs(i,0) + roo(0);
		ret(i, 1) = qs(i,1) + roo(1);
		ret(i, 2) = qs(i,2) + roo(2);
	}
	return ret;
}

/*
 * Sometimes we need vanilla state to ensure remeshing invariant
 *
 * R_{ompl} = R_{vanilla}
 * t_(ompl} = t_{vanilla} + R_{ompl} O
 * Hence,
 * t_{vanilla} = t_(ompl} - R_{ompl} O
 */
ArrayOfStates
UnitWorld::translateOMPLStateToVanillaState(const ArrayOfStates& qs) const
{
	static_assert(kStateDimension == 7, "Only support SE(3) for now");
	int N = qs.rows();
	ArrayOfStates van;
	van.resize(qs.rows(), qs.cols());
	// R_{ompl} = R_{vanilla}
	van.block(0, 3, N, 4) = qs.block(0, 3, N, 4);
	// Get OMPL center
	Eigen::Vector3d ompl_center = glm2Eigen(robot_->getOMPLCenter());
	for (int i = 0; i < N; i++) {
		// Vanilla state also uses w-last quaternion
		// TODO: Generalize to multi-body
		StateQuat rot(qs(i,6), qs(i,3), qs(i,4), qs(i,5));
		Eigen::Vector3d roo = rot.normalized()._transformVector(ompl_center);
		van(i, 0) = qs(i,0) - roo(0);
		van(i, 1) = qs(i,1) - roo(1);
		van(i, 2) = qs(i,2) - roo(2);
	}
	return van;
}

ArrayOfStates
UnitWorld::translateVanillaStateToUnitState(ArrayOfStates qs) const
{
	return translateOMPLStateToUnitState(translateVanillaStateToOMPLState(qs));
}

Eigen::MatrixXd
UnitWorld::translateVanillaPointsToUnitPoints(uint32_t geo,
                                              const Eigen::MatrixXd& pts) const
{
	Eigen::MatrixXd afpts;
	afpts.resize(pts.rows(), 4);
	if (geo == GEO_ENV) {
		afpts = pts.block(0, 0, pts.rows(), 3);
	} else {
		auto oc = robot_->getOMPLCenter();
		afpts = pts.block(0, 0, pts.rows(), 3).rowwise() - glm2Eigen(oc).transpose();
	}
	afpts.col(3) = Eigen::VectorXd::Constant(pts.rows(), 1.0);
	return (calib_mat_ * afpts.transpose()).transpose().block(0, 0, pts.rows(), 3);
}

ArrayOfStates
UnitWorld::translateOMPLStateToUnitState(ArrayOfStates qs) const
{
	int N = qs.rows();
	Eigen::Matrix<double, -1, 4> wlast(N, 4);
	wlast = qs.block(0, 3, N, 4);
	qs.col(3 + 0) = wlast.col(3);
	qs.col(3 + 1) = wlast.col(0);
	qs.col(3 + 2) = wlast.col(1);
	qs.col(3 + 3) = wlast.col(2);

	ArrayOfStates ret;
	ret.resize(qs.rows(), qs.cols());
	Eigen::Vector3d delta_center = glm2Eigen(robot_->getOMPLCenter() - robot_->getCenter());
	delta_center *= scene_scale_;
	for (int i = 0; i < N; i++) {
		ret.row(i) = translateToUnitState(qs.row(i));
		auto tup = decompose(ret.row(i).transpose());
		Eigen::Vector3d t_prime = std::get<0>(tup) - std::get<1>(tup) * delta_center;
		ret.row(i) = compose(t_prime, std::get<1>(tup));
	}
	return ret;
}

StateVector
UnitWorld::applyPertubation(const StateVector& state) const
{
	StateQuat rot, prot;
	StateTrans trans, ptrans;
	std::tie(trans, rot) = decompose(state);
	std::tie(ptrans, prot) = decompose(perturbate_);

	StateQuat qret = prot * rot;
	StateTrans tret = ptrans + prot.toRotationMatrix() * trans;
	return compose(tret, qret);
}

StateVector
UnitWorld::unapplyPertubation(const StateVector& state) const
{
	StateQuat rot, prot;
	StateTrans trans, ptrans;
	std::tie(trans, rot) = decompose(state);
	std::tie(ptrans, prot) = decompose(perturbate_);

	StateQuat qret = prot.inverse();
	StateTrans tret = qret.toRotationMatrix() * (-ptrans + trans);
	qret *= rot;
	return compose(tret, qret);
}

Eigen::Matrix<int8_t, -1, -1>
UnitWorld::calculateVisibilityMatrix(ArrayOfStates qs,
                                     bool is_unit_states,
                                     double verify_magnitude)
{
	int N = qs.rows();
	if (!is_unit_states) {
#pragma omp parallel for
		for (int i = 0; i < N; i++)
			qs.row(i) = translateToUnitState(qs.row(i)).transpose();
	}
	Eigen::Matrix<int8_t, -1, -1> ret;
	ret.resize(N, N);
#pragma omp parallel for
	for (int fi = 0; fi < N; fi++) {
		for (int ti = fi + 1; ti < N; ti++) {
			int8_t valid = !!isValidTransition(qs.row(fi), qs.row(ti), verify_magnitude);
			ret(fi, ti) = valid;
			ret(ti, fi) = valid;
		}
	}
	return ret;
}

Eigen::Matrix<int8_t, -1, -1>
UnitWorld::calculateVisibilityMatrix2(ArrayOfStates qs0,
                                      bool qs0_is_unit_states,
                                      ArrayOfStates qs1,
                                      bool qs1_is_unit_states,
                                      double verify_magnitude,
				      bool enable_mt)
{
	int M = qs0.rows();
	int N = qs1.rows();
	if (!qs0_is_unit_states)
#pragma omp parallel for if (enable_mt)
		for (int i = 0; i < M; i++)
			qs0.row(i) = translateToUnitState(qs0.row(i)).transpose();
	if (!qs1_is_unit_states)
#pragma omp parallel for if (enable_mt)
		for (int i = 0; i < N; i++)
			qs1.row(i) = translateToUnitState(qs1.row(i)).transpose();
	Eigen::Matrix<int8_t, -1, -1> ret;
	ret.resize(M, N);
	std::atomic<int> prog(0);
#pragma omp parallel for if (enable_mt)
	for (int fi = 0; fi < M; fi++) {
		for (int ti = 0; ti < N; ti++) {
			int8_t valid = isValidTransition(qs0.row(fi), qs1.row(ti), verify_magnitude);
			ret(fi, ti) = valid;
		}
		prog++;
		if (omp_get_thread_num() == 0) {
			std::cerr << "Progress: " << prog << "/" << M << std::endl;
		}
	}
	return ret;
}

#if PYOSR_HAS_CGAL
Eigen::Matrix<StateScalar, -1, 1>
UnitWorld::intersectionRegionSurfaceAreas(ArrayOfStates qs,
                                          bool qs_are_unit_states)
{
	ArrayOfStates qsu = ppToUnitStates(qs, qs_are_unit_states);
	int Nq = qs.rows();
	Eigen::Matrix<StateScalar, -1, 1> ret;
	ret.setZero(Nq);

	Transform envTf = std::get<0>(getCDTransforms(robot_state_));

	CDModel::VMatrix env_V = (envTf * cd_scene_->vertices().transpose()).transpose();
	auto env_F = cd_scene_->faces();

	auto rob_F = cd_robot_->faces();

	for (int i = 0; i < Nq; i++) {
		StateVector state = qsu.row(i).transpose();
		Transform robTf = translate_state_to_transform(state);
		CDModel::VMatrix rob_V = (robTf * cd_robot_->vertices().transpose()).transpose();

		CDModel::VMatrix RV;
		CDModel::FMatrix RF;
		mesh_bool(env_V, env_F,
			  rob_V, rob_F,
			  igl::MESH_BOOLEAN_TYPE_INTERSECT,
			  RV, RF);
		Eigen::VectorXd areas;
		igl::doublearea(RV, RF, areas);
		ret(i) = areas.sum();
	}
	return ret;
}

// FIXME: too much duplicated code here
std::tuple<UnitWorld::VMatrix, UnitWorld::FMatrix>
UnitWorld::intersectingGeometry(const StateVector& q,
                                bool q_is_unit)
{
	CDModel::VMatrix env_V, rob_V, RV;
	CDModel::FMatrix env_F, rob_F, RF;
	std::tie(env_V, env_F) = getSceneGeometry(q, q_is_unit);
#if 1
	std::tie(rob_V, rob_F) = getRobotGeometry(q, q_is_unit);
	mesh_bool(env_V, env_F,
	          rob_V, rob_F,
	          igl::MESH_BOOLEAN_TYPE_INTERSECT,
	          RV, RF);
	return std::make_tuple(RV, RF);
#endif
}

#endif // PYOSR_HAS_CGAL

std::tuple<UnitWorld::VMatrix, UnitWorld::FMatrix>
UnitWorld::getRobotGeometry(const StateVector& q,
                            bool q_is_unit) const
{
#if 1
	StateVector qu = q;
	if (!q_is_unit)
		qu = translateToUnitState(q);
	auto rob_F = cd_robot_->faces();
	Transform robTf = translate_state_to_transform(qu);
	CDModel::VMatrix rob_V = (robTf * cd_robot_->vertices().transpose()).transpose();

	return std::make_tuple(rob_V, rob_F);
#endif
}

std::tuple<UnitWorld::VMatrix, UnitWorld::FMatrix>
UnitWorld::getSceneGeometry(const StateVector& q,
                            bool q_is_unit) const
{
	StateVector qu = q;
	if (!q_is_unit)
		qu = translateToUnitState(q);
	Transform envTf = std::get<0>(getCDTransforms(qu));
	CDModel::VMatrix env_V = (envTf * cd_scene_->vertices().transpose()).transpose();
	auto env_F = cd_scene_->faces();
	return std::make_tuple(env_V, env_F);
}

std::tuple<UnitWorld::FMatrix, UnitWorld::VMatrix>
UnitWorld::intersectingToRobotSurface(const StateVector& q,
                                      bool q_is_unit,
                                      const UnitWorld::VMatrix& V,
                                      const UnitWorld::FMatrix& F)
{
	if (!robot_->hasUV()) {
		throw std::runtime_error("Calling intersectingToRobotSurface requires the corresponding model has UV coordinates");
	}

	CDModel::VMatrix rob_V;
	CDModel::FMatrix rob_F;
	std::tie(rob_V, rob_F) = getRobotGeometry(q, q_is_unit);
	return intersectingToSurface(rob_V, rob_F, V, F);
}

std::tuple<UnitWorld::FMatrix, UnitWorld::VMatrix>
UnitWorld::intersectingToModelSurface(const StateVector& q,
                                      bool q_is_unit,
                                      const UnitWorld::VMatrix& V,
                                      const UnitWorld::FMatrix& F)
{
	if (!scene_->hasUV()) {
		throw std::runtime_error("Calling intersectingToModelSurface requires the corresponding model has UV coordinates");
	}

	CDModel::VMatrix env_V;
	CDModel::FMatrix env_F;
	std::tie(env_V, env_F) = getSceneGeometry(q, q_is_unit);
	return intersectingToSurface(env_V, env_F, V, F);
}

std::tuple<
	ArrayOfPoints, // Position
	ArrayOfPoints, // Force vector
	Eigen::Matrix<StateScalar, -1, 1>,                // Force magnititude
	Eigen::Matrix<int, -1, 2>                         // Pairs of triangle indices
>
UnitWorld::intersectingSegments(StateVector unitq)
{
	ArrayOfPoints ret_pos, ret_vec;
	Eigen::Matrix<StateScalar, -1, 1> ret_mag;
	Eigen::Matrix<int, -1, 2> face_pairs;

	Transform env_tf;
	Transform rob_tf;
	std::tie(env_tf, rob_tf) = getCDTransforms(unitq);
	if (!CDModel::collideForDetails(*cd_scene_, env_tf, *cd_robot_, rob_tf, face_pairs)) {
		return std::tie(ret_pos, ret_vec, ret_mag, face_pairs);
	}
#if 0
	std::cerr << "debug: face pairs\n" << face_pairs << std::endl;
#endif
	CDModel::VMatrix env_V = (env_tf * cd_scene_->vertices().transpose()).transpose();
	CDModel::VMatrix rob_V = (rob_tf * cd_robot_->vertices().transpose()).transpose();
	auto env_F = cd_scene_->faces();
	auto rob_F = cd_robot_->faces();
	Eigen::Matrix<StateScalar, -1, 6> col_segs;
	Eigen::Matrix<int, -1, 1> coplanars;
	// NOTE: The (rob, env) order should be the same as calling
	// CDModel::collideForDetails
	tritri::TriTri(env_V, env_F,
	               rob_V, rob_F,
	               face_pairs,
	               col_segs, coplanars);
	int m = col_segs.rows();
	ret_pos = col_segs.block(0, 0, m, 3);
	ret_vec = col_segs.block(0, 3, m, 3);
	ret_mag = (ret_vec - ret_pos).array().square().rowwise().sum().sqrt().matrix(); // L2 norm per row
#if 0
	ret_pos = 0.5 * (col_segs.block(0, 0, m, 3) + col_segs.block(0, 3, m, 3));
	ret_vec = col_segs.block(0, 0, m, 3) - col_segs.block(0, 3, m, 3);
	for (int i = 0; i < m; i++) {
		// Face Normals (UNnormalized)
		Eigen::Vector3i rob_face = rob_F.row(face_pairs(i, 0));
		Eigen::Vector3i env_face = env_F.row(face_pairs(i, 0));
		Eigen::Vector3d rob_fn = (rob_V.row(rob_face(1)) - rob_V.row(rob_face(0))).cross(rob_V.row(rob_face(2)) - rob_V.row(rob_face(0)));
		Eigen::Vector3d env_fn = (env_V.row(env_face(1)) - env_V.row(env_face(0))).cross(env_V.row(env_face(2)) - env_V.row(env_face(0)));
		// Rational: force should be orthogonal w.r.t. intersecting
		// line segment and the robot's face normal (since robot is
		// movable)
		ret_vec.row(i) = rob_fn.cross(ret_vec.row(i)).normalized();
	}
#endif
	return std::tie(ret_pos, ret_vec, ret_mag, face_pairs);
}

ArrayOfPoints
UnitWorld::getRobotFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 1>& faces)
{
	return cd_robot_->faceNormals(faces);
}

ArrayOfPoints
UnitWorld::getRobotFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 2>& faces)
{
	return cd_robot_->faceNormals(faces.col(1));
}

ArrayOfPoints
UnitWorld::getSceneFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 1>& faces)
{
	return cd_scene_->faceNormals(faces);
}

ArrayOfPoints
UnitWorld::getSceneFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 2>& faces)
{
	return cd_scene_->faceNormals(faces.col(0));
}

std::tuple<
	ArrayOfPoints, // Force apply position
	ArrayOfPoints  // Force direction
>
UnitWorld::forceDirectionFromIntersectingSegments(
	const ArrayOfPoints& sbegins,
	const ArrayOfPoints& sends,
	const Eigen::Matrix<int, -1, 2> faces)
{
	ArrayOfPoints ret_pos, ret_dir;
	ret_pos = (sbegins + sends) * 0.5;
	ArrayOfPoints svec = sends - sbegins;
	igl::cross(svec, getRobotFaceNormalsFromIndices(faces), ret_dir);
	Eigen::Matrix<StateScalar, -1, 3> enormal = getSceneFaceNormalsFromIndices(faces);
	Eigen::Array<StateScalar, -1, 1> dots = (ret_dir.array() * enormal.array()).rowwise().sum().sign();

	ret_dir = ret_dir.array().colwise() * dots;

	return std::tie(ret_pos, ret_dir);
}

ArrayOfStates
UnitWorld::ppToUnitStates(const ArrayOfStates& qs,
                          bool qs_are_unit_states)
{
	if (qs_are_unit_states)
		return qs;
	ArrayOfStates qsu(qs.rows(), qs.cols());
	for (int i = 0; i < qs.rows(); i++)
		qsu.row(i) = translateToUnitState(qs.row(i));
	return qsu;
}

StateVector
UnitWorld::pushRobot(const StateVector& unitq,
                     const ArrayOfPoints& fpos,                     // Force apply position
                     const ArrayOfPoints& fdir,                     // Force direction
                     const Eigen::Matrix<StateScalar, -1, 1>& fmag, // Force magnititude
                     StateScalar mass,
                     StateScalar dtime,                             // Durition
                     bool resetVelocity
                    )
{
	if (!ode_) {
		ode_.reset(new OdeData(*cd_robot_));
	}
	if (resetVelocity)
		ode_->resetVelocity();
	ode_->setMass(mass, *cd_robot_);
	ode_->setState(unitq);
	ode_->applyForce(fpos, fdir, fmag);
	return ode_->stepping(dtime);
}

std::tuple<UnitWorld::FMatrix, UnitWorld::VMatrix>
UnitWorld::intersectingToSurface(const VMatrix& targetV,
                                 const FMatrix& targetF,
                                 const VMatrix& V,
                                 const FMatrix& F)
{
	Eigen::SparseMatrix<int> COP;
#if 0
	std::cerr << "TriTriCop between " << targetF.rows() << " " << F.rows() << std::endl;
#endif
	tritri::TriTriCopIsect(targetV, targetF, V, F, COP);
	size_t NF = COP.nonZeros();
#if 0
	std::cerr << "TriTriCop done; " << NF << " non-zeros\n";
#endif
	size_t NV = NF * 3;
	Eigen::Matrix<StateScalar, -1, 3> P(NV,3), A(NV,3), B(NV,3), C(NV,3), L(NV,3);
	// Eigen::Matrix<StateScalar, -1, 3> P, A, B, C, L;
	P.resize(NV, 3);
	A.resize(NV, 3);
	B.resize(NV, 3);
	C.resize(NV, 3);
	L.resize(NV, 3);
	FMatrix retF(NF, 3);
	size_t iter = 0;
#if 0
	std::cerr << "Bary build\n";
#endif
	for (int k = 0; k < COP.outerSize(); ++k) {
		for (Eigen::SparseMatrix<int>::InnerIterator it(COP, k); it; ++it) {
			auto target_fi = it.row();
			auto isect_fi = it.col();
#if 0
			std::cerr << "COP Pair " << target_fi << ", " << isect_fi << std::endl;
#endif
			P.row(3 * iter + 0) = V.row(F(isect_fi, 0));
			P.row(3 * iter + 1) = V.row(F(isect_fi, 1));
			P.row(3 * iter + 2) = V.row(F(isect_fi, 2));
			A.row(3 * iter + 0) = targetV.row(targetF(target_fi, 0));
			A.row(3 * iter + 1) = A.row(3 * iter + 0);
			A.row(3 * iter + 2) = A.row(3 * iter + 0);
			B.row(3 * iter + 0) = targetV.row(targetF(target_fi, 1));
			B.row(3 * iter + 1) = B.row(3 * iter + 0);
			B.row(3 * iter + 2) = B.row(3 * iter + 0);
			C.row(3 * iter + 0) = targetV.row(targetF(target_fi, 2));
			C.row(3 * iter + 1) = C.row(3 * iter + 0);
			C.row(3 * iter + 2) = C.row(3 * iter + 0);
			retF.row(iter) = targetF.row(target_fi);
			// std::cerr << iter << " done\n";
			iter++;
		}
	}
#if 0
	{
		Eigen::Matrix<int, -1, 3> F;
		F.resize(P.rows()/3, 3);
		for (int i = 0; i < P.rows() / 3; i++) {
			F.row(i) << i * 3 + 0,
			            i * 3 + 1,
			            i * 3 + 2;
		}
		igl::writeOBJ("tmp2.obj", P, F);
	}
	std::cerr << "Bary\n";
#endif
	igl::barycentric_coordinates(P, A, B, C, L);
#if 0
	// Sanity check
	for (int i = 0; i < P.rows(); i++) {
		Eigen::Vector3d tgt = A.row(i) * L(i,0) + B.row(i) * L(i,1) + C.row(i) * L(i,2);
		double d = (tgt - P.row(i).transpose()).norm();
		if (d > 1e-6) {
			std::cerr << "Bary failed, details:\n"
			          << "\tA: " << A.row(i) << "\n"
			          << "\tB: " << B.row(i) << "\n"
			          << "\tC: " << C.row(i) << "\n"
			          << "\tBary: " << L.row(i) << "\n"
			          << "\tP: " << P.row(i) << "\n";
			throw std::runtime_error("SANCHECK: BARYCENTRIC COORDINATES");
		}
	}
#endif

	return std::make_tuple(retF, L);
}

std::tuple<
	Eigen::Vector3d,                                // Position
	Eigen::Vector3d,                                // Normal
	Eigen::Vector2f                                 // UV
>
UnitWorld::sampleOverPrimitive(uint32_t geo_id,
                               int prim,
                               bool return_unit) const
{
	auto cd = getCDModel(geo_id);
	auto geo = getScene(geo_id);
	auto target_mesh = geo->getUniqueMesh();
	const auto& indices = target_mesh->getIndices();
	unsigned int vi[] = { indices[3 * prim + 0],
	                      indices[3 * prim + 1],
	                      indices[3 * prim + 2] };
	Eigen::Vector3d v[3];
	v[0] = cd->vertices().row(vi[0]);
	v[1] = cd->vertices().row(vi[1]);
	v[2] = cd->vertices().row(vi[2]);
	Eigen::Vector2f uv[3];
	uv[0] = target_mesh->getUV().row(vi[0]);
	uv[1] = target_mesh->getUV().row(vi[1]);
	uv[2] = target_mesh->getUV().row(vi[2]);

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 1.0);
#if 0
	double alpha = dis(gen);
	std::uniform_real_distribution<> dis2(0.0, 1.0 - alpha);
	double beta = dis2(gen);
	double gamma = 1.0 - alpha - beta;

#else
	// This is uniform sampling over the triangle face.
	double beta = dis(gen);
	double gamma = dis(gen);
	if (beta + gamma > 1.0) {
		beta = 1.0 - beta;
		gamma = 1.0 - gamma;
	}

	double alpha = 1.0 - beta - gamma;
	// Eigen::Vector3d interp_v = v[0] + u * (v[1] - v[0]) + v * (v[2] - v[0]);
	// Eigen::Vector2f interp_uv = uv[0] + u * (uv[1] - uv[0]) + v * (uv[2] - uv[0]);
#endif

	Eigen::Vector3d interp_v = alpha * v[0] + beta * v[1] + gamma * v[2];
	Eigen::Vector2f interp_uv = alpha * uv[0] + beta * uv[1] + gamma * uv[2];

	Eigen::Vector3d ret_pos = interp_v;
	Eigen::Vector3d ret_normal = cd->faceNormals().row(prim);
	if (!return_unit) {
#if 0 // Geometry stored in CDModel is already unit
		Transform tfs[2];
		StateVector identity_q;
		state_vector_set_identity(identity_q);
		std::tie(tfs[GEO_ENV], tfs[GEO_ROB]) = getCDTransforms(identity_q);

		ret_pos = tfs[geo_id] * ret_pos;
		ret_normal = tfs[geo_id].linear() * ret_normal;
#endif
		Eigen::Matrix4d tfmat = glm2Eigen(geo->getCalibrationTransform()).inverse();
		Transform tf(tfmat);
		ret_pos = tf * ret_pos;
		ret_normal = tf.linear() * ret_normal;
	}
	ret_normal.normalize();
	return std::make_tuple(ret_pos, ret_normal, interp_uv);
}


std::tuple<
	Eigen::Vector3d,                                // Position
	Eigen::Vector3d,                                // Normal
	bool
>
UnitWorld::uvToSurface(uint32_t geo_id,
                       int prim,
                       const Eigen::Vector2f& target_uv,
                       bool return_unit) const
{
	auto cd = getCDModel(geo_id);
	auto geo = getScene(geo_id);
	auto target_mesh = geo->getUniqueMesh();
	const auto& indices = target_mesh->getIndices();
	unsigned int vi[] = { indices[3 * prim + 0],
	                      indices[3 * prim + 1],
	                      indices[3 * prim + 2] };
	Eigen::Vector3d v[3];
	v[0] = cd->vertices().row(vi[0]);
	v[1] = cd->vertices().row(vi[1]);
	v[2] = cd->vertices().row(vi[2]);
	Eigen::Vector2f uv[3];
	uv[0] = target_mesh->getUV().row(vi[0]);
	uv[1] = target_mesh->getUV().row(vi[1]);
	uv[2] = target_mesh->getUV().row(vi[2]);
	//
	// target_uv = alpha * uv[0] + beta * uv[1] + gamma * uv[2]
	// alpha + beta + gamma = 1.0
	// , or
	// target_uv - uv[0] = beta * (uv[1] - uv[0]) + gamma * (uv[2] - uv[0])
	// alpha = 1.0 - beta - gamma
	Eigen::Matrix2f det;
	det.col(0) = uv[1] - uv[0];
	det.col(1) = uv[2] - uv[0];
	Eigen::Vector2f beta_gamma = det.inverse() * (target_uv - uv[0]);
	double beta = beta_gamma(0);
	double gamma = beta_gamma(1);
	double alpha = 1.0 - beta - gamma;
#if 0
	std::cerr << "UVs\n"
	          << uv[0].transpose() << "\n"
	          << uv[1].transpose() << "\n"
	          << uv[2].transpose() << "\n";
	std::cerr << "target_uv " << target_uv.transpose() << std::endl;
	std::cerr << "alpha beta gamma " << alpha << " " << beta << " " << gamma << std::endl;
	Eigen::Vector2f interp_uv = alpha * uv[0] + beta * uv[1] + gamma * uv[2];
	std::cerr << "target_uv err " << (target_uv - interp_uv).norm() << std::endl;
#endif
	bool valid = (std::min(gamma, std::min(alpha, beta)) >= 0.0);
	valid = valid && (std::max(gamma, std::max(alpha, beta)) <= 1.0);

	Eigen::Vector3d ret_pos = alpha * v[0] + beta * v[1] + gamma * v[2];
	Eigen::Vector3d ret_normal = cd->faceNormals().row(prim);
	if (!return_unit) {
		Eigen::Matrix4d tfmat = glm2Eigen(geo->getCalibrationTransform()).inverse();
		Transform tf(tfmat);
		ret_pos = tf * ret_pos;
		ret_normal = tf.linear() * ret_normal;
	}
	ret_normal.normalize();

	return std::make_tuple(ret_pos, ret_normal, valid);
}


StateVector
UnitWorld::sampleFreeConfiguration(const StateTrans& rob_surface_point,
                                   const StateTrans& rob_surface_normal,
                                   const StateTrans& env_surface_point,
                                   const StateTrans& env_surface_normal,
                                   StateScalar margin,
                                   int max_trials)
{
	StateTrans rob_o = rob_surface_point + rob_surface_normal * margin;
	StateTrans env_o = env_surface_point + env_surface_normal * margin;
	StateVector q; // return value

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, M_PI * 2);

	// Configuration (Q) sampling algorithm:
	// 1. Rotate Robot so that rob_surface_normal matches **negatived** env_surface_normal
	// 2. Rotate Robot with random angle axis (omega, env_surface_normal)
	// 3. Translate the rotated rob_surface_point to env_surface_point
	using Quat = Eigen::Quaternion<StateScalar>;
	using AA = Eigen::AngleAxis<StateScalar>;
	// Step 1 Rotation
	Quat rot_1;
	rot_1.setFromTwoVectors(rob_surface_normal, -env_surface_normal);
	int trials = 0;
	while (true) {
		// Step 2 Rotation
		Quat rot_2(AA(dis(gen), env_surface_normal));
		Quat rot_accum = rot_2 * rot_1;
		// Step 3 Translation
		StateTrans trans = env_o - (rot_accum * rob_o);
		q = compose(trans, rot_accum);
		if (isValid(q))
			break;
		if (max_trials >= 0) {
			trials += 1;
			if (trials > max_trials)
				break;
		}
	}
	return q;
}


ArrayOfStates
UnitWorld::enumFreeConfiguration(const StateTrans& rob_surface_point,
                                 const StateTrans& rob_surface_normal,
                                 const StateTrans& env_surface_point,
                                 const StateTrans& env_surface_normal,
                                 StateScalar margin,
                                 int denominator,
                                 bool only_median)
{
	StateTrans rob_o = rob_surface_point + rob_surface_normal * margin;
	StateTrans env_o = env_surface_point + env_surface_normal * margin;
	StateVector q; // return value

	// Configuration (Q) sampling algorithm:
	// 1. Rotate Robot so that rob_surface_normal matches **negatived** env_surface_normal
	// 2. Rotate Robot with random angle axis (omega, env_surface_normal)
	// 3. Translate the rotated rob_surface_point to env_surface_point
	using Quat = Eigen::Quaternion<StateScalar>;
	using AA = Eigen::AngleAxis<StateScalar>;
	// Step 1 Rotation
	Quat rot_1;
	rot_1.setFromTwoVectors(rob_surface_normal, -env_surface_normal);
	int trials = 0;
	double delta = 2 * M_PI / double(denominator);
	std::vector<StateVector> valid_states;
	std::vector<StateVector> current_valid_segment;
	for (int i = 0; i < denominator; i++) {
		// Step 2 Rotation
		Quat rot_2(AA(i * delta, env_surface_normal));
		Quat rot_accum = rot_2 * rot_1;

		// Step 3 Translation
		StateTrans trans = env_o - (rot_accum * rob_o);
		q = compose(trans, rot_accum);
		bool valid = isValid(q);
		if (!only_median) {
			if (valid)
				valid_states.emplace_back(q);
		} else {
			if (valid) {
				current_valid_segment.emplace_back(q);
			} else if (!current_valid_segment.empty()) {
				auto mid = current_valid_segment.begin() + current_valid_segment.size() / 2; 
				valid_states.emplace_back(*mid);
				current_valid_segment.clear();
			}
		}
	}
	ArrayOfStates ret;
	ret.resize(valid_states.size(), kStateDimension);
	for (size_t i = 0; i < valid_states.size(); i++) {
		ret.row(i) = valid_states[i];
	}
	return ret;
}

std::shared_ptr<Scene>
UnitWorld::getScene(uint32_t geo)
{
	if (geo == GEO_ENV) {
		return scene_;
	}
	if (geo == GEO_ROB) {
		return robot_;
	}
	throw std::runtime_error("Invalid geometry id: "+std::to_string(geo));
}

std::shared_ptr<CDModel>
UnitWorld::getCDModel(uint32_t geo)
{
	if (geo == GEO_ENV) {
		return cd_scene_;
	}
	if (geo == GEO_ROB) {
		return cd_robot_;
	}
	throw std::runtime_error("Invalid geometry id: "+std::to_string(geo));
}

std::shared_ptr<const Scene>
UnitWorld::getScene(uint32_t geo) const
{
	return const_cast<UnitWorld*>(this)->getScene(geo);
}

std::shared_ptr<const CDModel>
UnitWorld::getCDModel(uint32_t geo) const
{
	return const_cast<UnitWorld*>(this)->getCDModel(geo);
}

}
