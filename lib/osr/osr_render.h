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
#include <vector>

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
	static const uint32_t HAS_NTR_RENDERING = (1 << 2);
	static const uint32_t UV_MAPPINNG_RENDERING = (1 << 3);
	/*
	 * Define type for framebuffer attributes.
	 */
	typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> RMMatrixXf;
	typedef Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor> RMMatrixXb;
	typedef Eigen::Matrix<int32_t, -1, -1, Eigen::RowMajor> RMMatrixXi;

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
	void render_mvrgbd(uint32_t flags = 0);

	RMMatrixXb mvrgb;
	RMMatrixXf mvdepth;
	RMMatrixXf mvuv;
	RMMatrixXi mvpid;

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

	void setUVFeedback(bool);
	bool getUVFeedback() const;

	static const uint32_t BARY_RENDERING_ROBOT = 0;
	static const uint32_t BARY_RENDERING_SCENE = 1;

	void addBarycentric(const UnitWorld::FMatrix& F,
	                    const UnitWorld::VMatrix& V,
	                    uint32_t target);

	void clearBarycentric(uint32_t target);

	RMMatrixXb
	renderBarycentric(uint32_t target,
	                  Eigen::Vector2i res,
	                  const std::string& svg_fn = std::string());

	/*
	 * We may scale the whole scene as the final transformation
	 * within model transformation.
	 * 
	 * We expect this makes the NN insensitive to scaling.
	 */
	void setFinalScaling(const ScaleVector& scale);
	ScaleVector getFinalScaling() const;
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
	Camera setup_camera(uint32_t flags);

	glm::mat4 camera_rot_;

	GLuint uv_texture_ = 0;
	bool uvfeedback_enabled_ = false;
	void setupUVFeedbackBuffer();

	GLuint pid_texture_ = 0;
	void enablePidBuffer();

	/*
	 * Code to support barycentric rendering
	 */
	using BaryUV = Eigen::Matrix<float, -1, 2, Eigen::RowMajor>;
	using BaryBary = Eigen::Matrix<float, -1, 3, Eigen::RowMajor>;
	struct BaryRenderData {
		std::vector<BaryUV> uv_array;
		std::vector<BaryBary> bary_array;

		BaryUV cache_uv;
		BaryBary cache_bary;

		void sync(); // update cache
		void clear();
	};
	BaryRenderData brds_[2];
	std::shared_ptr<Scene> getBaryTarget(uint32_t);

	GLuint bary_texture_ = 0;
	GLuint bary_fb_ = 0;
	GLuint bary_dep_ = 0;
	GLuint bary_vs_ = 0;
	GLuint bary_gs_ = 0;
	GLuint bary_fs_ = 0;
	GLuint bary_shader_program_ = 0;
	GLuint bary_vao_ = 0;
	GLuint bary_vbo_uv_ = 0;
	GLuint bary_vbo_bary_ = 0;
	GLuint bary_ibo_ = 0;

	StateTrans final_scaling_;
};

}

#endif // GPU_ENABLED

#endif // OFF_SCREEN_RENDERING_TO_H
