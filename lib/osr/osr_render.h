#ifndef OFF_SCREEN_RENDERING_TO_H
#define OFF_SCREEN_RENDERING_TO_H

#include <Eigen/Core>
#include "quickgl.h"
#include <string>
#include <ostream>
#include <memory>

#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp>

namespace osr {
class Scene;
class Camera;

class Renderer {
public:
	typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> RMMatrixXf;

	Renderer();
	~Renderer();

	void setup();
	void teardown();
	void loadModelFromFile(const std::string& fn);
	void loadRobotFromFile(const std::string& fn);
	void scaleToUnit();
	void angleModel(float latitude, float longitude);
	void angleCamera(float latitude, float longitude);
	void render_depth_to(std::ostream& fout);
	Eigen::VectorXf render_depth_to_buffer();
	RMMatrixXf render_mvdepth_to_buffer();

	/*
	 * Accessors of Robot State. For rigid bodies, the state vector is:
	 *      Column 0-2 (x, y, z): translation vector
	 *      Column 3-6 (a, b, c, d): Quaternion a + bi + cj + dk for rotation.
	 */
	void setRobotState(const Eigen::VectorXf& state);
	Eigen::VectorXf getRobotState() const;

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
private:
	GLuint shaderProgram = 0;
	GLuint vertShader = 0;
	GLuint fragShader = 0;
	GLuint framebufferID = 0;
	GLuint depthbufferID = 0;
	GLuint renderTarget = 0;

	std::unique_ptr<Scene> scene_;
	std::unique_ptr<Scene> robot_;
	float scene_scale_ = 1.0f;
	Eigen::VectorXf robot_state_;
	glm::mat4 camera_rot_;

	void render_depth();
	Camera setup_camera();
};

}

#endif // OFF_SCREEN_RENDERING_TO_H
