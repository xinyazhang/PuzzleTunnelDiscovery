#if GPU_ENABLED

#include "osr_render.h"
#include "scene.h"
#include "scene_renderer.h"
#include "camera.h"
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/io.hpp>

namespace {
const char* vertShaderSrc =
#include "shader/default.vert"
;

const char* fragShaderSrc =
#include "shader/depth.frag"
;

const char* rgbdFragShaderSrc =
#include "shader/rgb.frag"
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

}

namespace osr {
const uint32_t Renderer::NO_SCENE_RENDERING;
const uint32_t Renderer::NO_ROBOT_RENDERING;
const uint32_t Renderer::HAS_NTR_RENDERING;
const uint32_t Renderer::UV_MAPPINNG_RENDERING;

Renderer::Renderer()
{
	camera_rot_ = glm::mat4(1.0f);
}

Renderer::~Renderer()
{
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
	GLuint fragShader = 0;
	GLuint rgbdFragShader = 0;

	/*
	 * vertex shader is shareable, but cannot be attached to multiple
	 * programs
	 */
	CHECK_GL_ERROR(vertShader = glCreateShader(GL_VERTEX_SHADER));
	CHECK_GL_ERROR(fragShader = glCreateShader(GL_FRAGMENT_SHADER));
	CHECK_GL_ERROR(rgbdFragShader = glCreateShader(GL_FRAGMENT_SHADER));
	int vlength = strlen(vertShaderSrc) + 1;
	int flength = strlen(fragShaderSrc) + 1;
	int flength2 = strlen(rgbdFragShaderSrc) + 1;
	CHECK_GL_ERROR(glShaderSource(vertShader, 1, &vertShaderSrc, &vlength));
	CHECK_GL_ERROR(glShaderSource(fragShader, 1, &fragShaderSrc, &flength));
	CHECK_GL_ERROR(glShaderSource(rgbdFragShader, 1, &rgbdFragShaderSrc, &flength2));
	CHECK_GL_ERROR(glCompileShader(vertShader));
	CHECK_GL_ERROR(glCompileShader(fragShader));
	CHECK_GL_ERROR(glCompileShader(rgbdFragShader));

	CheckShaderCompilation(vertShader);
	CheckShaderCompilation(fragShader);
	CheckShaderCompilation(rgbdFragShader);

	/*
	 * GLSL Shader Program is shareable, but it also tracks uniform
	 * bindings, so we need a copy rather than sharing directly.
	 */
	CHECK_GL_ERROR(shaderProgram = glCreateProgram());
	CHECK_GL_ERROR(glAttachShader(shaderProgram, vertShader));
	CHECK_GL_ERROR(glAttachShader(shaderProgram, fragShader));
	CHECK_GL_ERROR(glLinkProgram(shaderProgram));
	CheckProgramLinkage(shaderProgram);

	CHECK_GL_ERROR(rgbdShaderProgram = glCreateProgram());
	CHECK_GL_ERROR(glAttachShader(rgbdShaderProgram, vertShader));
	CHECK_GL_ERROR(glAttachShader(rgbdShaderProgram, rgbdFragShader));
	CHECK_GL_ERROR(glLinkProgram(rgbdShaderProgram));
	CheckProgramLinkage(rgbdShaderProgram);

	/*
	 * Delete non-used shaders to recycle resources.
	 */
	CHECK_GL_ERROR(glDeleteShader(vertShader));
	CHECK_GL_ERROR(glDeleteShader(fragShader));
	CHECK_GL_ERROR(glDeleteShader(rgbdFragShader));

	/*
	 * Create depth FB
	 */
	CHECK_GL_ERROR(glGenFramebuffers(1, &framebufferID));
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebufferID));
	CHECK_GL_ERROR(glGenTextures(1, &renderTarget));
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, renderTarget));
	CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, pbufferWidth, pbufferHeight, 0, GL_RED, GL_FLOAT, 0));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glGenRenderbuffers(1, &depthbufferID));
	CHECK_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, depthbufferID));
	CHECK_GL_ERROR(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, pbufferWidth, pbufferHeight));
	CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbufferID));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
	CHECK_GL_ERROR(glDrawBuffers(1, drawBuffers));

	/*
	 * Create RGBD FB
	 */
	CHECK_GL_ERROR(glGenFramebuffers(1, &rgbdFramebuffer));
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, rgbdFramebuffer));
	CHECK_GL_ERROR(glGenTextures(1, &rgbTarget));
	if (false /* No MSAA for now */) {
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, rgbTarget));
		CHECK_GL_ERROR(glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB8, pbufferWidth, pbufferHeight, false));
		CHECK_GL_ERROR(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D_MULTISAMPLE, rgbTarget, 0));
	} else {
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, rgbTarget));
		CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, pbufferWidth, pbufferHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rgbTarget, 0));
	}
	CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbufferID));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
	CHECK_GL_ERROR(glDrawBuffers(2, rgbdDrawBuffers));

	/*
	 * Switch back to depth-only FB
	 */
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebufferID));
}

void Renderer::teardown()
{
	CHECK_GL_ERROR(glDeleteFramebuffers(1, &framebufferID));
	CHECK_GL_ERROR(glDeleteRenderbuffers(1, &depthbufferID));
	CHECK_GL_ERROR(glDeleteTextures(1, &renderTarget));
	CHECK_GL_ERROR(glDeleteTextures(1, &rgbTarget));
	CHECK_GL_ERROR(glDeleteTextures(1, &uv_texture_));
	CHECK_GL_ERROR(glDeleteProgram(shaderProgram));
	CHECK_GL_ERROR(glDeleteProgram(rgbdShaderProgram));

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

	for(int i = 0; i < views.rows(); i++) {
		angleCamera(views(i, 0), views(i, 1));
		render_depth();
		CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RED, GL_FLOAT, mvpixels.row(i).data()));
	}
	return mvpixels;
}


void Renderer::render_mvrgbd(uint32_t flags)
{
	int is_render_uv_mapping = !!(flags & UV_MAPPINNG_RENDERING);

	setupUVFeedbackBuffer();
	if (is_render_uv_mapping) {
		enablePidBuffer();
	}

	mvrgb.resize(views.rows(), pbufferWidth * pbufferHeight * 3);
	mvdepth.resize(views.rows(), pbufferWidth * pbufferHeight);
	if (uvfeedback_enabled_)
		mvuv.resize(views.rows(), pbufferWidth * pbufferHeight * 2);
	else
		mvuv.resize(0, 0);

	for(int i = 0; i < views.rows(); i++) {
		angleCamera(views(i, 0), views(i, 1));
		render_rgbd(flags);
		CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RED, GL_FLOAT, mvdepth.row(i).data()));
		CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RGB, GL_UNSIGNED_BYTE, mvrgb.row(i).data()));
		if (uvfeedback_enabled_) {
			CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT2));
			CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
			                            GL_RG, GL_FLOAT, mvuv.row(i).data()));
		}
		if (is_render_uv_mapping) {
			CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT3));
			CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
			                            GL_RED_INTEGER, GL_INT, mvpid.row(i).data()));
		}
	}
}


void Renderer::render_depth()
{
	Camera camera = setup_camera(0);
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	CHECK_GL_ERROR(glClearTexImage(renderTarget, 0, GL_RED, GL_FLOAT, &default_depth));
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebufferID));

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
	int is_render_uv_mapping = !!(flags & UV_MAPPINNG_RENDERING);
#if 0
	std::cerr << "is_render_uv_mapping " << is_render_uv_mapping << "\n";
#endif

	glm::mat4 perturbation_mat = translate_state_to_matrix(perturbate_);
	Camera camera = setup_camera(flags);
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
#if 0
	static const uint8_t black[] = {0, 0, 255, 0};
	static const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};
#else
	static const uint8_t black[] = {0, 0, 0, 0};
#endif
	CHECK_GL_ERROR(glClearTexImage(rgbTarget, 0, GL_RGB, GL_UNSIGNED_BYTE, &black));
	CHECK_GL_ERROR(glClearTexImage(renderTarget, 0, GL_RED, GL_FLOAT, &default_depth));
	if (uv_texture_) {
		static const float invalid_uv[] = {-1.0f, -1.0f};
		CHECK_GL_ERROR(glClearTexImage(uv_texture_, 0, GL_RG, GL_FLOAT, invalid_uv));
	}
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, rgbdFramebuffer));
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
	CHECK_GL_ERROR(glUniform1i(16, is_render_uv_mapping ? 0 : avi));
	CHECK_GL_ERROR(glUniform3fv(17, 1, light_position.data()));
	CHECK_GL_ERROR(glUniform1i(18, 0));
	CHECK_GL_ERROR(glUniform1i(19, is_render_uv_mapping));

#if 1
	if (is_render_uv_mapping) {
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
		glEnable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
#endif

	if (!(flags & NO_SCENE_RENDERING)) {
		scene_renderer_->render(rgbdShaderProgram, camera, perturbation_mat, flags);
	}
	if (!(flags & NO_ROBOT_RENDERING) && robot_) {
		auto mat = translate_state_to_matrix(robot_state_);
		robot_renderer_->render(rgbdShaderProgram, camera, mat, flags);
	}
	CHECK_GL_ERROR(glUseProgram(0));

	CHECK_GL_ERROR(glFlush());
}


void Renderer::setUVFeedback(bool enable)
{
	uvfeedback_enabled_ = enable;

	/*
	 * Switch back to depth-only FB
	 */
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebufferID));
}


bool Renderer::getUVFeedback() const
{
	return uvfeedback_enabled_;
}


void Renderer::setupUVFeedbackBuffer()
{
	bool enable = uvfeedback_enabled_;
	if (uv_texture_ == 0 && enable) {
		CHECK_GL_ERROR(glGenTextures(1, &uv_texture_));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, uv_texture_));
		CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, pbufferWidth, pbufferHeight, 0, GL_RG, GL_FLOAT, 0));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
	}
	if (enable) {
		CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, rgbdFramebuffer));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rgbTarget, 0));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, uv_texture_, 0));
		CHECK_GL_ERROR(glDrawBuffers(3, rgbduvDrawBuffers));
	} else {
		CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, rgbdFramebuffer));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
		CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rgbTarget, 0));
		CHECK_GL_ERROR(glDrawBuffers(2, rgbdDrawBuffers));
	}
}


void Renderer::enablePidBuffer()
{
	if (!pid_texture_) {
		CHECK_GL_ERROR(glGenTextures(1, &pid_texture_));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, pid_texture_));
		CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, pbufferWidth, pbufferHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, 0));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
		CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, 0));
	}
	mvpid.resize(views.rows(), pbufferWidth * pbufferHeight);
	static const GLenum draw_buffers[3] = {
		GL_COLOR_ATTACHMENT0,
		GL_COLOR_ATTACHMENT1,
		// GL_COLOR_ATTACHMENT2, // UV_MAPPINNG_RENDERING disables uv_feedback
		GL_COLOR_ATTACHMENT3,
	};

	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, rgbdFramebuffer));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rgbTarget, 0));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, pid_texture_, 0));
	CHECK_GL_ERROR(glDrawBuffers(3, draw_buffers));

	// Switch back to depth-only FB
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, framebufferID));
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
	glm::vec4 eye = camera_rot_ * glm::vec4(0.0f, 0.0f, eyeDist, 1.0f);
	glm::vec4 cen = camera_rot_ * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
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
	return cam;
}

}

#endif // GPU_ENABLED
