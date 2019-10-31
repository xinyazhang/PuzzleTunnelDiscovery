#if GPU_ENABLED

#include "osr_render.h"
#include "scene.h"
#include "scene_renderer.h"
#include "camera.h"
#include "osr_rt_texture.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/io.hpp>
#include <fstream>

namespace {
const char* vertShaderSrc =
#include "shader/default.vert"
;

const char* geomShaderSrc =
#include "shader/default.geom"
;

const char* fragShaderSrc =
#include "shader/depth.frag"
;

const char* rgbdFragShaderSrc =
#include "shader/rgb.frag"
;

const char* bary_vs_src =
#include "shader/bary.vert"
;

const char* bary_gs_src =
#include "shader/bary.geom"
;

const char* bary_fs_src =
#include "shader/bary.frag"
;

const GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
const GLenum rgbdDrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
const GLenum rgbduvDrawBuffers[3] = { GL_COLOR_ATTACHMENT0,
	                              GL_COLOR_ATTACHMENT1,
				      GL_COLOR_ATTACHMENT2 };

using Renderer = osr::Renderer;
using StateScalar = osr::StateScalar;
using StateQuat = osr::StateQuat;
using StateTrans = osr::StateTrans;
using StateVector = osr::StateVector;
using StateMatrix = osr::StateMatrix;

glm::mat4 translate_state_to_matrix(const StateVector& state)
{
	glm::quat quat(state(3), state(4), state(5), state(6));
	glm::mat4 rot = glm::toMat4(quat);
	return glm::translate(glm::mat4(1.0f), glm::vec3(state(0), state(1), state(2))) * rot;
	// return glm::translate(glm::mat4(1.0f), glm::vec3(state(0), state(1), state(2)));
	// return rot;
}

template<typename T>
void make_with_default(std::shared_ptr<T>& ptr)
{
	ptr = std::make_shared<T>();
}

}

namespace osr {
const uint32_t Renderer::NO_SCENE_RENDERING;
const uint32_t Renderer::NO_ROBOT_RENDERING;
const uint32_t Renderer::HAS_NTR_RENDERING;
const uint32_t Renderer::UV_MAPPINNG_RENDERING;
const uint32_t Renderer::NORMAL_RENDERING;
const uint32_t Renderer::PID_RENDERING;

const uint32_t Renderer::BARY_RENDERING_ROBOT;
const uint32_t Renderer::BARY_RENDERING_SCENE;

Renderer::Renderer()
{
	camera_rot_ = glm::mat4(1.0f);
	final_scaling_ << 1.0, 1.0, 1.0;
}

Renderer::~Renderer()
{
	teardown();
}

void Renderer::setup()
{
	glGetError(); // Clear GL Error
	setupNonSharedObjects();
}

void Renderer::setupFrom(const Renderer* other)
{
	UnitWorld::copyFrom(other);

	setupNonSharedObjects();

	scene_renderer_.reset(new SceneRenderer(other->scene_renderer_));
	if (other->robot_) {
		robot_renderer_.reset(new SceneRenderer(other->robot_renderer_));
	} else {
		robot_renderer_.reset();
	}
}

void Renderer::setupNonSharedObjects()
{
	glGetError(); // Clear GL Error
	const GLubyte* renderer = glGetString(GL_RENDERER);  // get renderer string
	const GLubyte* version = glGetString(GL_VERSION);    // version as a string
	std::cout << "[" << __func__ <<  "] Renderer: " << renderer << "\n";
	std::cout << "[" << __func__ <<  "] OpenGL version supported:" << version << "\n";

	GLuint vertShader = 0;
	GLuint geomShader = 0;
	GLuint fragShader = 0;
	GLuint rgbdFragShader = 0;

	/*
	 * vertex shader is shareable, but cannot be attached to multiple
	 * programs
	 */
	CHECK_GL_ERROR(vertShader = glCreateShader(GL_VERTEX_SHADER));
	CHECK_GL_ERROR(geomShader = glCreateShader(GL_GEOMETRY_SHADER));
	CHECK_GL_ERROR(fragShader = glCreateShader(GL_FRAGMENT_SHADER));
	CHECK_GL_ERROR(rgbdFragShader = glCreateShader(GL_FRAGMENT_SHADER));
	int vlength = strlen(vertShaderSrc) + 1;
	int glength = strlen(geomShaderSrc) + 1;
	int flength = strlen(fragShaderSrc) + 1;
	int flength2 = strlen(rgbdFragShaderSrc) + 1;
	CHECK_GL_ERROR(glShaderSource(vertShader, 1, &vertShaderSrc, &vlength));
	CHECK_GL_ERROR(glShaderSource(geomShader, 1, &geomShaderSrc, &glength));
	CHECK_GL_ERROR(glShaderSource(fragShader, 1, &fragShaderSrc, &flength));
	CHECK_GL_ERROR(glShaderSource(rgbdFragShader, 1, &rgbdFragShaderSrc, &flength2));
	CHECK_GL_ERROR(glCompileShader(vertShader));
	CHECK_GL_ERROR(glCompileShader(geomShader));
	CHECK_GL_ERROR(glCompileShader(fragShader));
	CHECK_GL_ERROR(glCompileShader(rgbdFragShader));

	CheckShaderCompilation(vertShader);
	CheckShaderCompilation(geomShader);
	CheckShaderCompilation(fragShader);
	CheckShaderCompilation(rgbdFragShader);

	/*
	 * GLSL Shader Program is shareable, but it also tracks uniform
	 * bindings, so we need a copy rather than sharing directly.
	 */
	CHECK_GL_ERROR(shaderProgram = glCreateProgram());
	CHECK_GL_ERROR(glAttachShader(shaderProgram, vertShader));
	CHECK_GL_ERROR(glAttachShader(shaderProgram, geomShader));
	CHECK_GL_ERROR(glAttachShader(shaderProgram, fragShader));
	CHECK_GL_ERROR(glLinkProgram(shaderProgram));
	CheckProgramLinkage(shaderProgram);

	CHECK_GL_ERROR(rgbdShaderProgram = glCreateProgram());
	CHECK_GL_ERROR(glAttachShader(rgbdShaderProgram, vertShader));
	CHECK_GL_ERROR(glAttachShader(rgbdShaderProgram, geomShader));
	CHECK_GL_ERROR(glAttachShader(rgbdShaderProgram, rgbdFragShader));
	CHECK_GL_ERROR(glLinkProgram(rgbdShaderProgram));
	CheckProgramLinkage(rgbdShaderProgram);

	/*
	 * We still create barycentric rendering program here for early checks
	 */
	CHECK_GL_ERROR(bary_vs_ = glCreateShader(GL_VERTEX_SHADER));
	CHECK_GL_ERROR(bary_gs_ = glCreateShader(GL_GEOMETRY_SHADER));
	CHECK_GL_ERROR(bary_fs_ = glCreateShader(GL_FRAGMENT_SHADER));
	int bary_vs_len = strlen(bary_vs_src) + 1;
	int bary_gs_len = strlen(bary_gs_src) + 1;
	int bary_fs_len = strlen(bary_fs_src) + 1;
	CHECK_GL_ERROR(glShaderSource(bary_vs_, 1, &bary_vs_src, &bary_vs_len));
	CHECK_GL_ERROR(glShaderSource(bary_gs_, 1, &bary_gs_src, &bary_gs_len));
	CHECK_GL_ERROR(glShaderSource(bary_fs_, 1, &bary_fs_src, &bary_fs_len));
	CHECK_GL_ERROR(glCompileShader(bary_vs_));
	CHECK_GL_ERROR(glCompileShader(bary_gs_));
	CHECK_GL_ERROR(glCompileShader(bary_fs_));
	CHECK_GL_SHADER_ERROR(bary_vs_);
	CHECK_GL_SHADER_ERROR(bary_gs_);
	CHECK_GL_SHADER_ERROR(bary_fs_);

	CHECK_GL_ERROR(bary_shader_program_ = glCreateProgram());
	CHECK_GL_ERROR(glAttachShader(bary_shader_program_, bary_vs_));
	CHECK_GL_ERROR(glAttachShader(bary_shader_program_, bary_gs_));
	CHECK_GL_ERROR(glAttachShader(bary_shader_program_, bary_fs_));
	CHECK_GL_ERROR(glLinkProgram(bary_shader_program_));
	CHECK_GL_PROGRAM_ERROR(bary_shader_program_);
	CheckProgramLinkage(bary_shader_program_);

	/*
	 * Delete non-used shaders to recycle resources.
	 */
	CHECK_GL_ERROR(glDeleteShader(vertShader));
	CHECK_GL_ERROR(glDeleteShader(geomShader));
	CHECK_GL_ERROR(glDeleteShader(fragShader));
	CHECK_GL_ERROR(glDeleteShader(rgbdFragShader));

	/*
	 * Create depth FB
	 */
	make_with_default(depth_tex_);
	depth_tex_->ensure(pbufferWidth, pbufferHeight, 1, RtTexture::FLOAT_TYPE);
	make_with_default(depth_only_fb_);
	depth_only_fb_->attachRt(0, depth_tex_);
	depth_only_fb_->create(pbufferWidth, pbufferHeight);

	/*
	 * Create RGBD FB
	 */
	make_with_default(rgb_tex_);
	rgb_tex_->ensure(pbufferWidth, pbufferHeight, 3, RtTexture::BYTE_TYPE);
	make_with_default(rgbd_fb_);
	rgbd_fb_->attachRt(1, rgb_tex_);
	rgbd_fb_->attachRt(0, depth_tex_);
	rgbd_fb_->create(pbufferWidth, pbufferHeight);

	/*
	 * Switch back to depth-only FB
	 */
	depth_only_fb_->activate();

	make_with_default(uv_tex_);
	make_with_default(bary_tex_);
	make_with_default(bary_fb_);
	make_with_default(pid_tex_);
	make_with_default(normal_tex_);
}

void Renderer::teardown()
{
	depth_only_fb_.reset();
	rgbd_fb_.reset();
	depth_tex_.reset();
	rgb_tex_.reset();

	CHECK_GL_ERROR(glDeleteProgram(shaderProgram));
	CHECK_GL_ERROR(glDeleteProgram(rgbdShaderProgram));
	shaderProgram = 0;
	rgbdShaderProgram = 0;

	scene_.reset();
	robot_.reset();
	scene_renderer_.reset();
	robot_renderer_.reset();
}

void Renderer::loadModelFromFile(const std::string& fn)
{
	UnitWorld::loadModelFromFile(fn);
	scene_renderer_.reset(new SceneRenderer(scene_));
}

void Renderer::loadRobotFromFile(const std::string& fn)
{
	UnitWorld::loadRobotFromFile(fn);
	robot_renderer_.reset(new SceneRenderer(robot_));
	robot_renderer_->probe_texture(fn);
}

void Renderer::loadRobotTextureImage(std::string tex_fn)
{
	robot_renderer_->load_texture(tex_fn);
}

void Renderer::angleCamera(float latitude, float longitude)
{
	// camera_rot_ = glm::mat4(1.0f);
	camera_rot_ = glm::translate(glm::vec3(scene_->getCalibrationTransform() *
				     glm::vec4(scene_->getCenter(), 1.0)));
	// std::cerr << "Init camera_rot_\n"  << camera_rot_ << std::endl;
        camera_rot_ = glm::rotate(camera_rot_,
			glm::radians(latitude),
			glm::vec3(1.0f, 0.0f, 0.0f));
        camera_rot_ = glm::rotate(camera_rot_,
			glm::radians(longitude),
			glm::vec3(0.0f, 1.0f, 0.0f));
}

void Renderer::render_depth_to(std::ostream& fout)
{
	render_depth();

	std::vector<float> pixels(pbufferWidth * pbufferHeight);
	CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RED, GL_FLOAT, pixels.data()));

	const float *base = pixels.data();
	for (int i = 0; i < pbufferHeight; i++) {
		fout.write(reinterpret_cast<const char*>(base), sizeof(float) * pbufferWidth);
		base += pbufferWidth;
	}
}

Eigen::VectorXf Renderer::render_depth_to_buffer()
{
	Eigen::VectorXf pixels;
	render_depth();
	pixels.resize(pbufferWidth * pbufferHeight);
	CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RED,
				    GL_FLOAT, pixels.data()));
	return pixels;
}


/*
 * render_mvdepth_to_buffer
 *
 *     Render an object from multiple views into a matrix.
 */

Renderer::RMMatrixXf Renderer::render_mvdepth_to_buffer()
{
	RMMatrixXf mvpixels;
	mvpixels.resize(views.rows(), pbufferWidth * pbufferHeight);

	depth_only_fb_->activate();
	for(int i = 0; i < views.rows(); i++) {
		angleCamera(views(i, 0), views(i, 1));
		render_depth();

		depth_only_fb_->readShaderLocation(0);
		// CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RED, GL_FLOAT, mvpixels.row(i).data()));
	}
	return mvpixels;
}


void Renderer::render_mvrgbd(uint32_t flags)
{
	bool is_rendering_pid = !!(flags & PID_RENDERING);
	bool is_rendering_uv = !!(flags & UV_MAPPINNG_RENDERING);
	bool is_rendering_normal = !!(flags & NORMAL_RENDERING);
#if 0
	std::cerr << "is_rendering_uv " << is_rendering_uv << std::endl;
#endif

	rgbd_fb_->attachRt(0, depth_tex_);
	rgbd_fb_->attachRt(1, rgb_tex_);
	if (is_rendering_uv) {
		mvuv.resize(views.rows(), pbufferWidth * pbufferHeight * 2);
		uv_tex_->ensure(pbufferWidth, pbufferHeight, 2, RtTexture::FLOAT_TYPE);
		rgbd_fb_->attachRt(2, uv_tex_);
	} else {
		rgbd_fb_->detachRt(2);
		mvuv.resize(0, 0);
	}
	if (is_rendering_pid) {
		mvpid.resize(views.rows(), pbufferWidth * pbufferHeight);
		pid_tex_->ensure(pbufferWidth, pbufferHeight, 1, RtTexture::INT32_TYPE);
		rgbd_fb_->attachRt(3, pid_tex_);
	}
	if (is_rendering_normal) {
		mvnormal.resize(views.rows(), pbufferWidth * pbufferHeight * 3);
		normal_tex_->ensure(pbufferWidth, pbufferWidth, 3, RtTexture::FLOAT_TYPE);
		rgbd_fb_->attachRt(4, normal_tex_);
	}
	rgbd_fb_->activate();

	mvrgb.resize(views.rows(), pbufferWidth * pbufferHeight * 3);
	mvdepth.resize(views.rows(), pbufferWidth * pbufferHeight);

	for(int i = 0; i < views.rows(); i++) {
		angleCamera(views(i, 0), views(i, 1));
		render_rgbd(flags);
		rgbd_fb_->readShaderLocation(0);
		// CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RED, GL_FLOAT, mvdepth.row(i).data()));
		rgbd_fb_->readShaderLocation(1);
		// CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RGB, GL_UNSIGNED_BYTE, mvrgb.row(i).data()));
		if (is_rendering_uv) {
			rgbd_fb_->readShaderLocation(2);
			// CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT2));
			CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
			                            GL_RG, GL_FLOAT, mvuv.row(i).data()));
		}
		if (is_rendering_pid) {
#if 1
			rgbd_fb_->readShaderLocation(3);
			// CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT3));
			CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
			                            GL_RED_INTEGER, GL_INT, mvpid.row(i).data()));
#else
			CHECK_GL_ERROR(glGetTextureImage(uv_texture_, 0, GL_RED_INTEGER, GL_INT, mvpid.cols() * sizeof(int32_t), mvpid.row(i).data()));
#endif
		}
		if (is_rendering_normal) {
			rgbd_fb_->readShaderLocation(4);
			CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
			                            GL_RGB, GL_FLOAT, mvnormal.row(i).data()));
		}
	}
}


void Renderer::render_depth()
{
	Camera camera = setup_camera(0);
	depth_only_fb_->deactivate();
	depth_tex_->clear(&default_depth);
	depth_only_fb_->activate();

	CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
	CHECK_GL_ERROR(glDepthFunc(GL_LESS));
	CHECK_GL_ERROR(glViewport(0, 0, pbufferWidth, pbufferHeight));
	CHECK_GL_ERROR(glClearColor(0.0, 0.0, 0.0, 0.0));
	// CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	CHECK_GL_ERROR(glClear(GL_DEPTH_BUFFER_BIT));

	CHECK_GL_ERROR(glUseProgram(shaderProgram));
	scene_renderer_->render(shaderProgram, camera, glm::mat4(1.0), 0);
	if (robot_) {
		auto mat = translate_state_to_matrix(robot_state_);
		robot_renderer_->render(shaderProgram, camera, mat, 0);
	}
	CHECK_GL_ERROR(glUseProgram(0));

	CHECK_GL_ERROR(glFlush());
}

void Renderer::render_rgbd(uint32_t flags)
{
	int is_rendering_uv = !!(flags & UV_MAPPINNG_RENDERING);
#if 0
	std::cerr << "is_rendering_uv " << is_rendering_uv << "\n";
#endif

	glm::mat4 perturbation_mat = translate_state_to_matrix(perturbate_);
	Camera camera = setup_camera(flags);
	rgbd_fb_->deactivate();
	depth_tex_->clear();
	rgb_tex_->clear();
	pid_tex_->clear();
	static const float invalid_uv[] = {-1.0f, -1.0f};
	uv_tex_->clear(invalid_uv);
	static const int invalid_pid[] = {-1};
	pid_tex_->clear(invalid_pid);
	normal_tex_->clear();
	rgbd_fb_->activate();
#if 0
	CHECK_GL_ERROR(glFlush());
	return;
#endif

	CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
	CHECK_GL_ERROR(glDepthFunc(GL_LESS));
	CHECK_GL_ERROR(glViewport(0, 0, pbufferWidth, pbufferHeight));
	// CHECK_GL_ERROR(glClearColor(0.0, 0.0, 0.0, 0.0));
	// CHECK_GL_ERROR(glClearBufferfv(GL_COLOR, GL_DRAW_BUFFER0, zeros));
	// CHECK_GL_ERROR(glClearBufferfv(GL_COLOR, GL_DRAW_BUFFER1, zeros));
	// CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
	CHECK_GL_ERROR(glClear(GL_DEPTH_BUFFER_BIT));

	CHECK_GL_ERROR(glUseProgram(rgbdShaderProgram));
	// AVI is disabled when rendering UV
	CHECK_GL_ERROR(glUniform1i(16, is_rendering_uv ? 0 : avi));
	CHECK_GL_ERROR(glUniform3fv(17, 1, light_position.data()));
	CHECK_GL_ERROR(glUniform1i(18, 0));
	CHECK_GL_ERROR(glUniform1i(19, is_rendering_uv));

#if 0
	if (is_rendering_uv) {
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glEnable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
#else
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Always fill, otherwise mvpid will not be correct
#endif

	if (!(flags & NO_SCENE_RENDERING)) {
		/*
		 * Using flat surface normal generated by GS if
		 * 1. User set flat_surface to true
		 * 2. Vertex normal is missing in the geometry
		 */
		bool flat;
		if (scene_->hasVertexNormal())
			flat = flat_surface;
		else
			flat = true;
		CHECK_GL_ERROR(glUniform1i(20, flat));
		scene_renderer_->render(rgbdShaderProgram, camera, perturbation_mat, flags);
	}
	if (!(flags & NO_ROBOT_RENDERING) && robot_) {
		bool flat;
		if (robot_->hasVertexNormal())
			flat = flat_surface;
		else
			flat = true;
		CHECK_GL_ERROR(glUniform1i(20, flat));
		auto mat = translate_state_to_matrix(robot_state_);
		robot_renderer_->render(rgbdShaderProgram, camera, mat, flags);
	}
	CHECK_GL_ERROR(glUseProgram(0));

	CHECK_GL_ERROR(glFlush());
}


std::shared_ptr<Scene>
Renderer::getBaryTarget(uint32_t target)
{
	std::shared_ptr<Scene> target_scene;
	if (target == BARY_RENDERING_ROBOT)
		target_scene = robot_;
	else if (target == BARY_RENDERING_SCENE)
		target_scene = scene_;
	else
		throw std::runtime_error(std::string(__func__) + ": unknow target " + std::to_string(target));
	if (!target_scene->hasUV())
		throw std::runtime_error(std::string(__func__) + ": target mesh has no UV coordinates");
	return target_scene;
}

void
Renderer::addBarycentric(const UnitWorld::FMatrix& F,
                         const UnitWorld::VMatrix& V,
                         uint32_t target,
                         float weight)
{
	auto target_scene = getBaryTarget(target);
	auto target_mesh = target_scene->getUniqueMesh();

#if 0
	const Mesh *target_mesh = nullptr;
	auto visitor = [&target_mesh](std::shared_ptr<const Mesh> m) {
		target_mesh = m.get();
	};
	target_scene->visitMesh(visitor);
	if (!target_mesh)
		throw std::runtime_error(std::string(__func__) + ": Target has no valid mesh");
#endif
	const auto& tex_uv = target_mesh->getUV();
	// Assembly the UV and Bary

	size_t NP = F.rows(); // Number of primitives (presumably a small number)
	BaryUV uv(NP * 3, 2);
	for (size_t f = 0; f < NP; f++) {
		for (size_t i = 0; i < 3; i++) {
			auto f_bary = F(f, i);
			uv.row(3 * f + i) = tex_uv.row(f_bary);
		}
	}
	brds_[target].uv_array.emplace_back(uv);
	brds_[target].bary_array.emplace_back(V.cast<float>());
	brds_[target].weight_array.emplace_back(BaryWeight::Constant(1, V.rows(), weight));
}

void
Renderer::clearBarycentric(uint32_t target)
{
	auto target_scene = getBaryTarget(target); // Ensure target is valid
	brds_[target].clear();

	if (bary_vao_ != 0) {
		// Clear graphics memory
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_uv_));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_bary_));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_weight_));
		CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, 0, NULL, GL_STATIC_DRAW));
		CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));
	}
}

Renderer::RMMatrixXf
Renderer::renderBarycentric(uint32_t target,
                            Eigen::Vector2i res,
                            const std::string& svg_fn)
{
	auto target_scene = getBaryTarget(target);

	bary_tex_->ensure(res(0), res(1), 1, RtTexture::FLOAT_TYPE);
	bary_fb_->attachRt(0, bary_tex_);
	bary_fb_->create(pbufferWidth, pbufferHeight);

	bary_tex_->clear();
	bary_fb_->activate();

	CHECK_GL_ERROR(glDisable(GL_DEPTH_TEST));
	CHECK_GL_ERROR(glDisable(GL_CULL_FACE));
	CHECK_GL_ERROR(glPolygonMode(GL_FRONT_AND_BACK, GL_FILL));
	CHECK_GL_ERROR(glViewport(0, 0, res(0), res(1)));
	// CHECK_GL_ERROR(glViewport(0, 0, 1024, 1024));
	// CHECK_GL_ERROR(glViewport(0, 0, 256, 256));

	CHECK_GL_ERROR(glClear(GL_DEPTH_BUFFER_BIT));

	CHECK_GL_ERROR(glUseProgram(bary_shader_program_));
#if 0
	Camera cam;
	glm::vec4 eye = glm::vec4(0.0f, 0.0f, -1.5, 1.0f);
	glm::vec4 cen = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 upv = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	cam.lookAt(
			glm::vec3(eye),     // eye
			glm::vec3(cen),     // CENter
			glm::vec3(upv)      // UP Vector
	          );
	cam.perspective(
	                glm::radians(45.0f),                           // fov
	                1.0,                                            // aspect ratio
	                0.01,                                       // near plane
	                1000.0f                                         // far plane
			);
	cam.uniform(bary_shader_program_, glm::mat4(1.0));
#endif

	// Prepare the data
	if (!bary_vao_) {
		CHECK_GL_ERROR(glGenVertexArrays(1, &bary_vao_));
		CHECK_GL_ERROR(glGenBuffers(1, &bary_vbo_uv_));
		CHECK_GL_ERROR(glGenBuffers(1, &bary_vbo_bary_));
		CHECK_GL_ERROR(glGenBuffers(1, &bary_vbo_weight_));
#if 0
		CHECK_GL_ERROR(glGenBuffers(1, &bary_ibo_));
#endif
	}

	CHECK_GL_ERROR(glBindVertexArray(bary_vao_));
	BaryRenderData& brd = brds_[target];
#if 1
	brd.sync();
#else
	{
		auto target_scene = getBaryTarget(target);

		const Mesh *target_mesh = nullptr;
		auto visitor = [&target_mesh](std::shared_ptr<const Mesh> m) {
			target_mesh = m.get();
		};
		target_scene->visitMesh(visitor);
		if (!target_mesh)
			throw std::runtime_error(std::string(__func__) + ": Target has no valid mesh");
		const auto& tex_uv = target_mesh->getUV();
		const auto& F = target_mesh->getIndices();
		size_t NP = F.size() / 3;
		BaryUV &uv = brd.cache_uv;
		BaryBary &cache_bary = brd.cache_bary;

		uv.resize(NP * 3, 2);
		cache_bary.resize(NP * 3, 3);
		for (size_t f = 0; f < NP; f++) {
			for (size_t i = 0; i < 3; i++) {
				auto f_bary = F[f * 3 + i];
				uv.row(3 * f + i) = tex_uv.row(f_bary);
			}
			cache_bary.row(3 * f + 0) << 1.0, 0.0, 0.0;
			cache_bary.row(3 * f + 1) << 0.0, 1.0, 0.0;
			cache_bary.row(3 * f + 2) << 0.0, 0.0, 1.0;
		}
	}
#endif

	if (!svg_fn.empty()) {
		std::ofstream fout(svg_fn);
		fout << "<svg width=\"" << res(0) << "\" height=\"" << res(1) << "\">\n";
		for (int i = 0; i < brd.cache_uv.rows(); i+=3) {
			using Eigen::Vector2f;
			Vector2f u0 = brd.cache_uv.row(i + 0);
			Vector2f u1 = brd.cache_uv.row(i + 1);
			Vector2f u2 = brd.cache_uv.row(i + 2);

			Vector2f uvs[3];
			for (int k = 0; k < 3; k++) {
				uvs[k] = u0 * brd.cache_bary(i + k,0) + u1 * brd.cache_bary(i + k,1) + u2 * brd.cache_bary(i + k,2);
#if 0
				std::cerr << "Bary " << i+k << ":\n\t"
					  << uvs[k].transpose()
					  << std::endl;
#endif
				// brd.cache_uv.row(i + k) = uvs[k];
				uvs[k](0) *= res(0);
				uvs[k](1) *= res(1);
			}
			fout << R"xxx(<polygon points=")xxx"
			     << uvs[0](0) <<',' << uvs[0](1) << " "
			     << uvs[1](0) <<',' << uvs[1](1) << " "
			     << uvs[2](0) <<',' << uvs[2](1) << " "
			     << R"xxx(" style="fill:lime;stroke:purple;stroke-width:1" />)xxx"
			     << std::endl;
		}
		fout << R"xxx(</svg>)xxx";
	}

#if 1
	CHECK_GL_ERROR(glEnableVertexAttribArray(0));
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_uv_));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
	                            brd.cache_uv.size() * sizeof(float),
	                            brd.cache_uv.data(), GL_STATIC_DRAW));
	// std::cerr << "glBufferData " << brd.cache_uv.size() << " * " << sizeof(float) << std::endl;
	CHECK_GL_ERROR(glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
					     0, 0));
#endif
#if 1
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_bary_));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
	                            brd.cache_bary.size() * sizeof(float),
	                            brd.cache_bary.data(), GL_STATIC_DRAW));
	CHECK_GL_ERROR(glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
					     0, 0));
	CHECK_GL_ERROR(glEnableVertexAttribArray(1));
#endif
	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, bary_vbo_weight_));
	CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER,
	                            brd.cache_weight.size() * sizeof(float),
	                            brd.cache_weight.data(), GL_STATIC_DRAW));
	CHECK_GL_ERROR(glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE,
					     0, 0));
	CHECK_GL_ERROR(glEnableVertexAttribArray(2));

	CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, 0));

#if 0
	const uint32_t index_data[] = {0, 1, 2};
	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bary_ibo_));
	CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bary_ibo_));
	CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER,
	                            sizeof(uint32_t) * 3,
	                            index_data, GL_STATIC_DRAW));
	CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0));
#else
	CHECK_GL_ERROR(glEnable(GL_BLEND));
	CHECK_GL_ERROR(glBlendEquation(GL_FUNC_ADD));
	CHECK_GL_ERROR(glBlendFunc(GL_ONE, GL_ONE));
	CHECK_GL_ERROR(glDrawArrays(GL_TRIANGLES, 0, brd.cache_uv.rows()));
	CHECK_GL_ERROR(glDisable(GL_BLEND));
#endif
	CHECK_GL_ERROR(glFinish());
	CHECK_GL_ERROR(glBindVertexArray(0));

	RMMatrixXf pixels(res(0), res(1));
	CHECK_GL_ERROR(glReadPixels(0, 0, res(0), res(1), GL_RED, GL_FLOAT, pixels.data()));

	return pixels;
}

Camera Renderer::setup_camera(uint32_t flags)
{
#if 0
	if (flags & UV_MAPPINNG_RENDERING) {
		Camera cam;
		cam.lookAt(glm::vec3(0.5, 0.5, -1.0),
			   glm::vec3(0.5, 0.5, 0.0),
			   glm::vec3(0.0, 1.0, 0.0));
		// cam.ortho2D(0.0, 1.0, 0.0, 1.0);
		// cam.ortho(-5, 5, -5, 5, -5, 5);
		cam.ortho(-0.5, 0.5, -0.5, 0.5, -5, 5);
		// cam.ortho2D(-2, 2, -2, 2);
		// cam.ortho2D(-5, 5, -5, 5);
		return cam;
	}
#endif
	const float eyeDist = 2.0f;
	const float minDist = 0.01f;
	Camera cam;
	glm::vec4 cen = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	glm::vec4 eye = cen + camera_rot_ * glm::vec4(0.0f, 0.0f, eyeDist, 0.0f);
	glm::vec4 upv = camera_rot_ * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	cam.lookAt(
			glm::vec3(eye),     // eye
			glm::vec3(cen),     // CENter
			glm::vec3(upv)      // UP Vector
	          );
	cam.perspective(
			glm::radians(45.0f),                           // fov
			(float) pbufferWidth / (float) pbufferHeight,  // aspect ratio
			minDist,                                       // near plane
			120.0f                                         // far plane
			);
	cam.scale(final_scaling_[0], final_scaling_[1], final_scaling_[2]);
	return cam;
}

void Renderer::BaryRenderData::sync()
{
	size_t total = 0;
#if 1
	for (const auto& uv : uv_array)
		total += uv.rows();
	cache_uv.resize(total, 2);
	cache_bary.resize(total, 3);
	cache_weight.resize(1, total);
	size_t cursor = 0;
	for (size_t i = 0; i < uv_array.size(); i++) {
		cache_uv.block(cursor, 0, uv_array[i].rows(), 2) = uv_array[i];
		cache_bary.block(cursor, 0, bary_array[i].rows(), 3) = bary_array[i];
		cache_weight.block(0, cursor, weight_array[i].cols(), 1) = weight_array[i];
		cursor += uv_array[i].rows();
	}
#else
#if 0
	cache_uv.resize(3, 2);
	cache_bary.resize(3, 3);
	cache_uv << 0.25,0.25,
	            0.25,0.75,
	            0.75,0.25;
#if 0
	cache_bary << 1.0, 0.0, 0.0,
	              0.0, 1.0, 0.0,
	              0.0, 0.0, 1.0;
#else
	cache_bary << -0.2, 0.6, 0.6,
	              0.6, -0.2, 0.6,
	              0.6, 0.6, -0.2;
#endif
#elif 0
	cache_bary << -1.0f, -1.0f, 0.0f,
	              1.0f, -1.0f, 0.0f,
	              0.0f,  1.0f, 0.0f;
	cache_bary = cache_bary.array() * 0.5;
#else
#endif
#endif

#if 0
	std::cerr << "CACHE UV\n" << cache_uv << std::endl;
	std::cerr << "CACHE BARY\n" << cache_bary << std::endl;
#endif
}

void Renderer::BaryRenderData::clear()
{
	uv_array.clear();
	bary_array.clear();
	weight_array.clear();
}


void Renderer::setFinalScaling(const ScaleVector& scale)
{
	final_scaling_ = scale;
}

ScaleVector Renderer::getFinalScaling() const
{
	return final_scaling_;
}

}

#endif // GPU_ENABLED
