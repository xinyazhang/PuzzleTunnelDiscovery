#ifndef OFF_SCREEN_RENDERING_TO_H
#define OFF_SCREEN_RENDERING_TO_H

#include <Eigen/Core>
#include "quickgl.h"
#include <string>
#include <ostream>
#include <memory>
#include <stdint.h>
#include <tuple>

#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp>

namespace osr {
class Scene;
class Camera;
class CDModel;

/*
 * osr::Renderer
 *
 *      This class loads robot and environment model, and render them to
 *      buffers. At the same time, it is also responsible for collision
 *      detection (CD).
 *
 *      Only support rigid body for now.
 *
 *      TODO: We may want a better name because CD is also handled by this
 *      class.
 */
class Renderer {
public:
	typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> RMMatrixXf;
	typedef Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor> RMMatrixXb;

	typedef double StateScalar;
	typedef Eigen::Vector3d StateTrans;
	typedef Eigen::Quaternion<double> StateQuat;
	typedef Eigen::VectorXd StateVector;
	typedef Eigen::Matrix4d StateMatrix;

	Renderer();
	~Renderer();

	void setup();
	void setupFrom(const Renderer*);
	void teardown();
	void loadModelFromFile(const std::string& fn);
	void loadRobotFromFile(const std::string& fn);
	void scaleToUnit();
	void angleModel(float latitude, float longitude);
	void angleCamera(float latitude, float longitude);
	void render_depth_to(std::ostream& fout);
	Eigen::VectorXf render_depth_to_buffer();
	RMMatrixXf render_mvdepth_to_buffer();
	void render_mvrgbd();

	RMMatrixXb mvrgb;
	RMMatrixXf mvdepth;
	/*
	 * Accessors of Robot State. For rigid bodies, the state vector is:
	 *      Column 0-2 (x, y, z): translation vector
	 *      Column 3-6 (a, b, c, d): Quaternion a + bi + cj + dk for rotation.
	 */
	void setRobotState(const StateVector& state);
	StateVector getRobotState() const;

	bool isValid(const StateVector& state) const;
	bool isDisentangled(const StateVector& state) const;

	int pbufferWidth = 224;
	int pbufferHeight = 224;
	float default_depth = 5.0f;
	/*
	 * Set this variable to use multple view rendering.
	 * Protocol:
	 *      Row K: View K
	 *        Column 0: latitude;
	 *        Column 1: longitude
	 */
	Eigen::MatrixXf views;
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
	             Eigen::Vector2f magnitudes,
	             Eigen::Vector2f deltas) const;

	StateMatrix getSceneMatrix() const;
	StateMatrix getRobotMatrix() const;

	/*
	 * Interpolate states
	 */
	static StateVector interpolate(const StateVector& pkey,
	                               const StateVector& nkey,
	                               StateScalar tau);
	/*
	 * Translate from unscale and uncentralized world coordinates to the
	 * unit cube coordinates
	 */
	StateVector translateToUnitState(const StateVector& state);
private:
	void setupNonSharedObjects();
	GLuint shaderProgram = 0;
	GLuint rgbdShaderProgram = 0;
	GLuint framebufferID = 0;
	GLuint depthbufferID = 0;
	GLuint renderTarget = 0;
	GLuint rgbdFramebuffer = 0;
	GLuint rgbTarget = 0;

	bool shared_ = false;

	std::shared_ptr<Scene> scene_;
	std::shared_ptr<Scene> robot_;
	std::shared_ptr<CDModel> cd_scene_;
	std::shared_ptr<CDModel> cd_robot_;
	float scene_scale_ = 1.0f;
	StateVector robot_state_;
	glm::mat4 camera_rot_;

	void render_depth();
	void render_rgbd();
	Camera setup_camera();
};

constexpr int kActionDimension = 3;
/* *2 for pos/neg */
constexpr int kActionPerTransformType = kActionDimension * 2;
/* *2 for translation/rotation */
constexpr int kTotalNumberOfActions = kActionPerTransformType * 2;

/* 3 for Translation, 4 for quaternion in SO(3) */
constexpr int kStateDimension = 3 + 4;
}

#endif // OFF_SCREEN_RENDERING_TO_H
