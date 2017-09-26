#include "osr_render.h"
#include "scene.h"
#include "camera.h"
#include <glm/gtx/quaternion.hpp>

namespace {
const char* vertShaderSrc =
R"zzz(
#version 330
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
out vec3 fragColor;
out gl_PerVertex {
    vec4 gl_Position;
};
out float linearZ;
void main() {
    vec4 vm = view * model * vec4(inPosition, 1.0);
    gl_Position = proj * vm;
    linearZ = length(vec3(vm));
    fragColor = inColor;
}
)zzz";

const char* fragShaderSrc =
R"zzz(
#version 330
in vec3 fragColor;
in float linearZ;
// layout(location=0) out vec4 outColor;
layout(location=0) out float outDepth;
out vec4 outColor;
const float far = 20.0;
const float near = 1.0;
void main() {
    // gl_FragDepth = (1.0 / gl_FragCoord.w - near) / (far - near);
    // outColor = vec4(fragColor, 1.0);
    outDepth = linearZ;
}
)zzz";

const char* rgbdFragShaderSrc =
R"zzz(
#version 330
in vec3 fragColor;
in float linearZ;
layout(location=0) out float outDepth;
layout(location=1) out vec4 outColor;
const float far = 20.0;
const float near = 1.0;
void main() {
    outColor = vec4(fragColor, 1.0);
    outDepth = linearZ;
}
)zzz";

const GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
const GLenum rgbdDrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

glm::mat4 translate_state_to_matrix(const Eigen::VectorXf& state)
{
	glm::quat quat(state(3), state(4), state(5), state(6));
	glm::mat4 rot = glm::toMat4(quat);
	return glm::translate(rot, glm::vec3(state(0), state(1), state(2)));
}

}

namespace osr {
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
	const GLubyte* renderer = glGetString(GL_RENDERER);  // get renderer string
	const GLubyte* version = glGetString(GL_VERSION);    // version as a string
	std::cout << "Renderer: " << renderer << "\n";
	std::cout << "OpenGL version supported:" << version << "\n";

	setupNonSharedObjects();
}

void Renderer::setupFrom(const Renderer* other)
{
	shared_ = true;

	setupNonSharedObjects();

	scene_.reset(new Scene(other->scene_));
	if (other->robot_)
		robot_.reset(new Scene(other->robot_));
	else
		robot_.reset();
}

void Renderer::setupNonSharedObjects()
{
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
	CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, rgbTarget));
	CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, pbufferWidth, pbufferHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	CHECK_GL_ERROR(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthbufferID));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderTarget, 0));
	CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, rgbTarget, 0));
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
	CHECK_GL_ERROR(glDeleteProgram(shaderProgram));
	CHECK_GL_ERROR(glDeleteProgram(rgbdShaderProgram));

	scene_.reset();
}

void Renderer::loadModelFromFile(const std::string& fn)
{
	scene_.reset(new Scene);
	const glm::vec3 blue(0.0f, 0.0f, 1.0f);
	scene_->load(fn, &blue);
}

void Renderer::loadRobotFromFile(const std::string& fn)
{
	robot_.reset(new Scene);
	const glm::vec3 red(1.0f, 0.0f, 0.0f);
	robot_->load(fn, &red);
	robot_state_.setZero(7);
	robot_state_(3) = 1.0; // Quaternion for no rotation
}

void Renderer::scaleToUnit()
{
	auto scene_span =  scene_->getBoundingBox().span();
	float robot_span = 1.0f;
	if (robot_)
		robot_span =  robot_->getBoundingBox().span();
	scene_scale_ = 1.0 / std::max(scene_span, robot_span);
}

void Renderer::angleModel(float latitude, float longitude)
{
	scene_->resetTransform();
	scene_->scale(glm::vec3(scene_scale_));
	scene_->rotate(glm::radians(latitude), 1, 0, 0);      // latitude
	scene_->rotate(glm::radians(longitude), 0, 1, 0);     // longitude
	scene_->moveToCenter();
	if (robot_) {
		robot_->resetTransform();
		robot_->scale(glm::vec3(scene_scale_));
		robot_->rotate(glm::radians(latitude), 1, 0, 0);      // latitude
		robot_->rotate(glm::radians(longitude), 0, 1, 0);     // longitude
		robot_->moveToCenter();
	}
}

void Renderer::angleCamera(float latitude, float longitude)
{
	camera_rot_ = glm::mat4(1.0f);
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


void Renderer::render_mvrgbd()
{
	mvrgb.resize(views.rows(), pbufferWidth * pbufferHeight * 3);
	mvdepth.resize(views.rows(), pbufferWidth * pbufferHeight);

	for(int i = 0; i < views.rows(); i++) {
		angleCamera(views(i, 0), views(i, 1));
		render_rgbd();
		CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT0));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RED, GL_FLOAT, mvdepth.row(i).data()));
		CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));
		CHECK_GL_ERROR(glReadPixels(0, 0, pbufferWidth, pbufferHeight,
					    GL_RGB, GL_UNSIGNED_BYTE, mvrgb.row(i).data()));
	}
}


void Renderer::setRobotState(const Eigen::VectorXf& state)
{
	robot_state_ = state;
}

Eigen::VectorXf Renderer::getRobotState() const
{
	return robot_state_;
}

void Renderer::render_depth()
{
	Camera camera = setup_camera();
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
	scene_->render(shaderProgram, camera, glm::mat4());
	if (robot_) {
		auto mat = translate_state_to_matrix(robot_state_);
		robot_->render(shaderProgram, camera, mat);
	}
	CHECK_GL_ERROR(glUseProgram(0));

	CHECK_GL_ERROR(glFlush());
}

void Renderer::render_rgbd()
{
	Camera camera = setup_camera();
	CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
#if 0
	static const uint8_t black[] = {0, 0, 255, 0};
#else
	static const uint8_t black[] = {0, 0, 0, 0};
#endif
	static const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};
	CHECK_GL_ERROR(glClearTexImage(rgbTarget, 0, GL_RGB, GL_UNSIGNED_BYTE, &black));
	CHECK_GL_ERROR(glClearTexImage(renderTarget, 0, GL_RED, GL_FLOAT, &default_depth));
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
	scene_->render(rgbdShaderProgram, camera, glm::mat4());
	if (robot_) {
		auto mat = translate_state_to_matrix(robot_state_);
		robot_->render(rgbdShaderProgram, camera, mat);
	}
	CHECK_GL_ERROR(glUseProgram(0));

	CHECK_GL_ERROR(glFlush());
}

Camera Renderer::setup_camera()
{
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
