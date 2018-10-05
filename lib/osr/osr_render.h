#ifndef OFF_SCREEN_RENDERING_TO_H
#define OFF_SCREEN_RENDERING_TO_H

#if GPU_ENABLED

#include "osr_state.h"
#include "unit_world.h"
#include "quickgl.h"
#include <string>
#include <ostream>
#include <memory>
#include <stdint.h>
#include <tuple>

#define GLM_FORCE_RADIANS
#include <glm/mat4x4.hpp>

namespace osr {

class SceneRenderer;
class Camera;

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
class Renderer : public UnitWorld {
public:
	static const uint32_t NO_SCENE_RENDERING = (1 << 0);
	static const uint32_t NO_ROBOT_RENDERING = (1 << 1);
	/*
	 * Define type for framebuffer attributes.
	 */
	typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> RMMatrixXf;
	typedef Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor> RMMatrixXb;

	Renderer();
	~Renderer();

	void setup();
	void setupFrom(const Renderer*);
	void loadModelFromFile(const std::string& fn) override;
	void loadRobotFromFile(const std::string& fn) override;
	void teardown();
	void angleCamera(float latitude, float longitude);

	void render_depth_to(std::ostream& fout);
	Eigen::VectorXf render_depth_to_buffer();
	RMMatrixXf render_mvdepth_to_buffer();
	void render_mvrgbd(uint32_t flags);

	RMMatrixXb mvrgb;
	RMMatrixXf mvdepth;

	int pbufferWidth = 224;
	int pbufferHeight = 224;
	float default_depth = 5.0f;
	GLboolean avi = false; // AdVanced Illumination
	Eigen::Vector3f light_position = {0.0f, 5.0f, 0.0f};

	/*
	 * Set this variable to use multple view rendering.
	 * Protocol:
	 *      Row K: View K
	 *        Column 0: latitude;
	 *        Column 1: longitude
	 */
	Eigen::MatrixXf views;
	
	Eigen::MatrixXf getPermutationToWorld(int view);
private:
	void setupNonSharedObjects();

	GLuint shaderProgram = 0;
	GLuint rgbdShaderProgram = 0;
	GLuint framebufferID = 0;
	GLuint depthbufferID = 0;
	GLuint renderTarget = 0;
	GLuint rgbdFramebuffer = 0;
	GLuint rgbTarget = 0;

	std::shared_ptr<SceneRenderer> scene_renderer_;
	std::shared_ptr<SceneRenderer> robot_renderer_;

	void render_depth();
	void render_rgbd(uint32_t flags);
	Camera setup_camera();

	glm::mat4 camera_rot_;
};

}

#endif // GPU_ENABLED

#endif // OFF_SCREEN_RENDERING_TO_H
