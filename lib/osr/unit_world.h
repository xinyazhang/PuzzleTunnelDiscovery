#ifndef OSR_UNIT_WORLD_H
#define OSR_UNIT_WORLD_H

#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp>

#include <memory>
#include <tuple>
#include "osr_state.h"

namespace osr {
class Scene;
class CDModel;

class UnitWorld {
public:
	UnitWorld();
	~UnitWorld();

	void copyFrom(const UnitWorld*);
	virtual void loadModelFromFile(const std::string& fn);
	virtual void loadRobotFromFile(const std::string& fn);
	void scaleToUnit();
	void angleModel(float latitude, float longitude);

	/*
	 * Accessors of Robot State. For rigid bodies, the state vector is:
	 *      Column 0-2 (x, y, z): translation vector
	 *      Column 3-6 (a, b, c, d): Quaternion a + bi + cj + dk for rotation.
	 */
	void setRobotState(const StateVector& state);
	StateVector getRobotState() const;

	std::tuple<Transform, Transform> getCDTransforms(const StateVector& state) const;
	bool isValid(const StateVector& state) const;
	bool isDisentangled(const StateVector& state) const;

	/*
	 * State transition
	 *
	 *      state: initial state
	 *      action: which action (translate/rotate w.r.t +-XYZ axes) to
	 *              take
	 *              odd: positive action
	 *              even: negative action
	 *              0-5: translation w.r.t. XYZ
	 *              6-11: rotation w.r.t. XYZ
	 *      magnitudes: magnitudes of transition vector and radius of rotation
	 *      deltas: stepping magnitudes for discrete CD, (0) for
	 *              translation, (1) for rotation.
	 *
	 * Return value: tuple of (new state, is final, progress)
	 *      new state: indicate the last valid state
	 *      is final: indicate if the return state finished the expected action.
	 *      progress: 0.0 - 1.0, the ratio b/w the finished action and the
	 *                expected action. progress < 1.0 implies is final == False
	 * 
	 * TODO: change deltas to double, to align with distance defined in
	 * GTGenerator (or MTBlender)
	 */
	std::tuple<StateVector, bool, float>
	transitState(const StateVector& state,
	             int action,
                     double transit_magnitude,
                     double verify_delta) const;

	std::tuple<StateVector, bool, float>
	transitStateTo(const StateVector& from,
	               const StateVector& to,
	               double magnitude) const;

	StateMatrix getSceneMatrix() const;
	StateMatrix getRobotMatrix() const;

	/*
	 * Translate from/to unscale and uncentralized world coordinates
	 * to/from the unit cube coordinates
	 */
	StateVector translateToUnitState(const StateVector& state);
	StateVector translateFromUnitState(const StateVector& state);
protected:
	bool shared_ = false;

	std::shared_ptr<Scene> scene_;
	std::shared_ptr<Scene> robot_;

	std::shared_ptr<CDModel> cd_scene_;
	std::shared_ptr<CDModel> cd_robot_;

	StateVector robot_state_;
	glm::mat4 the_world_;
	Eigen::Matrix4d calib_mat_, inv_calib_mat_;
	Eigen::Vector3d world_rebase_;
	float scene_scale_ = 1.0f;
};

auto glm2Eigen(const glm::mat4& m);

}

#endif
