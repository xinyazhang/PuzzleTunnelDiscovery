/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OSR_UNIT_WORLD_H
#define OSR_UNIT_WORLD_H

#include <glm/mat4x4.hpp>

#include <memory>
#include <tuple>
#include "osr_state.h"
#include <stdint.h>

namespace osr {
class Scene;
class CDModel;
struct OdeData;

class UnitWorld {
public:
	UnitWorld();
	~UnitWorld();

	static const uint32_t GEO_ENV = 0;
	static const uint32_t GEO_ROB = 1;

	void copyFrom(const UnitWorld*);
	// Model, also known as Scene Geometry (commonly used in Renderer),
	// or Environment Geometry (abbr. into "env" in Python code)
	virtual void loadModelFromFile(const std::string& fn);
	virtual void loadRobotFromFile(const std::string& fn);
	void enforceRobotCenter(const StateTrans&);
	void scaleToUnit();
	void angleModel(float latitude, float longitude);

	/*
	 * The 'state' of the Env Geomtry.
	 */
	void setPerturbation(const StateVector& pert);
	StateVector getPerturbation() const;

	/*
	 * Accessors of Robot State. For rigid bodies, the state vector is:
	 *      Column 0-2 (x, y, z): translation vector
	 *      Column 3-6 (a, b, c, d): Quaternion a + bi + cj + dk for rotation.
	 */
	void setRobotState(const StateVector& state);
	StateVector getRobotState() const;

	StateTrans getModelCenter() const;

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
	 */
	std::tuple<StateVector, bool, float>
	transitState(const StateVector& state,
	             int action,
                     double transit_magnitude,
                     double verify_delta) const;

	// Returns:
	//      0: last free configuration
	//      1: is trajectory collision free
	//      2: free/total length of trajectory
	// Note: this function returns (from, false, 0.0) if verify_delta >= |from - to|
	std::tuple<StateVector, bool, float>
	transitStateTo(const StateVector& from,
	               const StateVector& to,
	               double verify_delta) const;

	// Returns:
	//      0: last_free configuration
	//      1: first colliding configuration
	//      2: is trajectory collision free
	//      3: free/total length of trajectory
	//      4: first colliding/total length of trajectory
	// Note: this function returns
	//      a) (from, to, false, 0.0, 1.0) if verify_delta >= |from - to|
	//      b) (to, to, true, 1.0, 1.0) if from -- to is collision-free.
	//           b.1) so do not use <1> without checking <2>
	std::tuple<StateVector, StateVector, bool, float, float>
	transitStateToWithContact(const StateVector& from,
	                          const StateVector& to,
	                          double verify_delta) const;

	bool
	isValidTransition(const StateVector& from,
	                  const StateVector& to,
	                  double initial_verify_delta) const;

	std::tuple<StateVector, bool, float>
	transitStateBy(const StateVector& from,
	               const StateTrans& tr,
	               const AngleAxisVector& aa,
	               double verify_delta) const;

	StateMatrix getSceneMatrix() const;
	StateMatrix getRobotMatrix() const;

	/*
	 * Translate from/to unscale and uncentralized world coordinates
	 * to/from the unit cube coordinates
	 *
	 * Note: do NOT pass OMPL states to these two functions.
	 *       1. They do NOT handle the translation between w-first and w-last.
	 *       2. The reference point used by these functions may be
	 *          overriden by enforceRobotCenter, and hence may not align
	 *          with the OMPL reference point
	 */
	StateVector translateToUnitState(const StateVector& state) const;
	StateVector translateFromUnitState(const StateVector& state) const;
	// Translate an array of unit cube coordinate states to OMPL state
	// Note: this is different from translateFromUnitState since OMPL has
	//       fixed center, while our center can be overrided by
	//       enforceRobotCenter.
	ArrayOfStates translateUnitStateToOMPLState(const ArrayOfStates& qs,
	                                            bool to_angle_axis = false) const;
	// Note: do not pass by const& because the implementation does need a copy
	ArrayOfStates translateOMPLStateToUnitState(ArrayOfStates qs) const;
	// Translate the OMPL state to Vanillay State, which is the direct
	// translation from the geometry's own coordinate system.
	// Vanilla state also uses w-last format
	ArrayOfStates translateVanillaStateToOMPLState(const ArrayOfStates& qs) const;
	ArrayOfStates translateOMPLStateToVanillaState(const ArrayOfStates& qs) const;

	ArrayOfStates translateVanillaStateToUnitState(ArrayOfStates qs) const;

	Eigen::MatrixXd translateVanillaPointsToUnitPoints(uint32_t geo,
							   const Eigen::MatrixXd& pts) const;

	StateVector applyPertubation(const StateVector& state) const;
	StateVector unapplyPertubation(const StateVector& state) const;

	/*
	 * Tunnel Finder support function: visibility matrix calculator
	 */
	Eigen::Matrix<int8_t, -1, -1>
	calculateVisibilityMatrix(ArrayOfStates qs,
	                          bool is_unit_states,
	                          double verify_magnitude);
	Eigen::Matrix<int8_t, -1, -1>
	calculateVisibilityMatrix2(ArrayOfStates qs0,
	                           bool qs0_is_unit_states,
	                           ArrayOfStates qs1,
	                           bool qs1_is_unit_states,
	                           double verify_magnitude,
				   bool enable_mt = true);
	Eigen::Matrix<int8_t, -1, 1>
	calculateVisibilityPair(ArrayOfStates qs0,
			        bool qs0_is_unit_states,
	                        ArrayOfStates qs1,
	                        bool qs1_is_unit_states,
	                        double verify_magnitude,
				bool enable_mt = true);

	using VMatrix = Eigen::Matrix<StateScalar, -1, 3>;
	using FMatrix = Eigen::Matrix<int, -1, 3>;

#if PYOSR_HAS_MESHBOOL
	Eigen::Matrix<StateScalar, -1, 1>
	intersectionRegionSurfaceAreas(ArrayOfStates qs,
	                               bool qs_are_unit_states);

	std::tuple<VMatrix, FMatrix>
	intersectingGeometry(const StateVector& q,
	                     bool q_is_unit);
#endif
#if 1
	std::tuple<VMatrix, FMatrix>
	getRobotGeometry(const StateVector& q,
	                 bool q_is_unit) const;
#endif

	std::tuple<VMatrix, FMatrix>
	getSceneGeometry(const StateVector& q,
	                 bool q_is_unit) const;
	/*
	 * Requirements:
	 *      UV coordintates must present
	 *
	 * Return:
	 *      1) List of face indices in the Robot geometry
	 *      2) barycentric coordinates in the robot geometry face.
	 */
	std::tuple<FMatrix, VMatrix>
	intersectingToRobotSurface(const StateVector& q,
	                           bool q_is_unit,
	                           const VMatrix& V,
	                           const FMatrix& F);

	// Similar to intersectingToRobotSurface
	std::tuple<FMatrix, VMatrix>
	intersectingToModelSurface(const StateVector& q,
	                           bool q_is_unit,
	                           const VMatrix& V,
	                           const FMatrix& F);

	std::tuple<
		ArrayOfPoints, // Segment beginnings
		ArrayOfPoints, // Segment ends
		Eigen::Matrix<StateScalar, -1, 1>,                // Segment magnititudes
		Eigen::Matrix<int, -1, 2>                         // (env, rob) face indices
	>
	intersectingSegments(StateVector unitq);

	ArrayOfPoints
	getRobotFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 1>&);
	ArrayOfPoints
	getRobotFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 2>&);

	ArrayOfPoints
	getSceneFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 1>&);
	ArrayOfPoints
	getSceneFaceNormalsFromIndices(const Eigen::Matrix<int, -1, 2>&);

	std::tuple<
		ArrayOfPoints, // Force apply position
		ArrayOfPoints  // Force direction
	>
	forceDirectionFromIntersectingSegments(
		const ArrayOfPoints& sbegins,
		const ArrayOfPoints& sends,
		const Eigen::Matrix<int, -1, 2> faces);

	StateVector
	pushRobot(const StateVector& unitq,
	          const ArrayOfPoints& fpos,                     // Force apply position
	          const ArrayOfPoints& fdir,                     // Force direction
	          const Eigen::Matrix<StateScalar, -1, 1>& fmag, // Force magnititude
	          StateScalar mass,
	          StateScalar dtime,                             // Durition
	          bool resetVelocity                             // Reset the stored velocity
	         );

	std::tuple<
		Eigen::Vector3d,                                // Position
		Eigen::Vector3d,                                // Normal
		Eigen::Vector2f                                 // UV
	>
	sampleOverPrimitive(uint32_t geo,
	                    int prim,
	                    bool return_unit = true) const;

	std::tuple<
		Eigen::Vector3d,                                // Position
		Eigen::Vector3d,                                // Normal
		bool                                            // Valid in Prim
	>
	uvToSurface(uint32_t geo,
	            int prim,
	            const Eigen::Vector2f& uv,
	            bool return_unit = true) const;

	//
	// Sample a free configuration from the surface information
	// Points and normals shall be those returned by sampleOverPrimitive,
	// with return_unit == true
	//
	// Parameters:
	//      free_guarantee: set to false to just sample a configuration
	StateVector
	sampleFreeConfiguration(const StateTrans& rob_surface_point,
	                        const StateTrans& rob_surface_normal,
	                        const StateTrans& env_surface_point,
	                        const StateTrans& env_surface_normal,
	                        StateScalar margin,
	                        int max_trials = -1);

	// Like sampleFreeConfiguration, only accepts vectors of unit states
	ArrayOfStates
	enumFreeConfiguration(const StateTrans& rob_surface_point,
	                      const StateTrans& rob_surface_normal,
	                      const StateTrans& env_surface_point,
	                      const StateTrans& env_surface_normal,
	                      StateScalar margin,
	                      int denominator,
	                      bool only_median = false);

	ArrayOfStates
	enum2DRotationFreeConfiguration(const StateTrans& rob_surface_point,
	                                const StateTrans& rob_surface_normal,
	                                const int rob_prim_id,
	                                const StateTrans& env_surface_point,
	                                const StateTrans& env_surface_normal,
	                                const int env_prim_id,
				        StateScalar init_margin,
				        int azimuth_divider,
				        int altitude_divider,
				        bool return_all = false,
				        bool enable_mt = true);


	std::shared_ptr<Scene> getScene(uint32_t geo);
	std::shared_ptr<CDModel> getCDModel(uint32_t geo);

	std::shared_ptr<const Scene> getScene(uint32_t geo) const;
	std::shared_ptr<const CDModel> getCDModel(uint32_t geo) const;

	// Get/Set Recommended Collision (checking) Resolution
	void
	setRecommendedCres(double res)
	{
		recCres_ = res;
	}

	double
	getRecommendedCres() const
	{
		return recCres_;
	}

	double
	kineticEnergyDistance(const StateVector& q0,
	                      const StateVector& q1) const;

	Eigen::VectorXd
	multiKineticEnergyDistance(const StateVector& origin,
	                           const ArrayOfStates& targets);

	double getSceneScale() const
	{
		return scene_scale_;
	}

	Eigen::Vector3d getOMPLCenter(uint32_t geo = GEO_ROB) const;
protected:
	bool shared_ = false;

	std::shared_ptr<Scene> scene_;
	std::shared_ptr<Scene> robot_;

	std::shared_ptr<CDModel> cd_scene_;
	std::shared_ptr<CDModel> cd_robot_;

	StateVector robot_state_;
	Eigen::Matrix4d calib_mat_, inv_calib_mat_;
	StateVector perturbate_;
	Transform perturbate_tf_;
	Eigen::Vector3d world_rebase_;
	float scene_scale_ = 1.0f;

	// pp: PreProcess
	ArrayOfStates ppToUnitStates(const ArrayOfStates& qs,
	                             bool qs_are_unit_states);

	std::unique_ptr<OdeData> ode_;

	std::tuple<FMatrix, VMatrix>
	intersectingToSurface(const VMatrix& targetV,
	                      const FMatrix& targetF,
	                      const VMatrix& V,
	                      const FMatrix& F);

	void
	extractTriangle(uint32_t geo_id,
	                int prim,
	                Eigen::Vector3d v[3],
	                Eigen::Vector2f uv[3],
	                Eigen::Vector3d *fn) const;

	double recCres_;
};

auto glm2Eigen(const glm::mat4& m);

}

#endif
