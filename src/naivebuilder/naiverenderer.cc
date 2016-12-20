#include "naiverenderer.h"
#include <renderer/quickgl.h>
#include <renderer/render_pass.h>
#include <renderer/gui.h>
#include <omplaux/geo.h>
#include <mutex>
#include <thread>

struct Naive2DRenderer::Private {
	GLFWwindow *window;
	std::unique_ptr<GUI> *gui;
	const Geo* env;
	std::unique_ptr<std::thread> worker;
	std::mutex mutex;

	Private()
	{
		mutex->lock();
	}

};

Naive2DRenderer::Naive2DRenderer()
	:p_(new Naive2DRenderer::Private)
{
}

Naive2DRenderer::~Naive2DRenderer()C
{
}

// TODO: Tree structure.
void Naive2DRenderer::addSplit(const Eigen::VectorXd& center, 
		      const Eigen::VectorXd& mins,
		      const Eigen::VectorXd& maxs)
{
}

void Naive2DRenderer::addCertain(const Eigen::VectorXd& center, 
			const Eigen::VectorXd& mins,
			const Eigen::VectorXd& maxs)
{
}

void Naive2DRenderer::init()
{
	p_->window = init_glefw(800, 600, "Naive Renderer");
	p_->gui.reset(new GUI(window));
}

void Naive2DRenderer::launch_worker(std::function<int(NaiveRenderer*)> fn)
{
	p_->worker.reset(new std::thread(fn, this));
}

void Naive2DRenderer::setEnv(const Geo* env) override
{
	p_->env = env;
}

void Naive2DRenderer::workerReady()
{
	mutex.unlock();
}

const char* vertex_shader =
#include "shaders/default.vert"
;

const char* geometry_shader =
#include "shaders/default.geom"
;

const char* fragment_shader =
#include "shaders/default.frag"
;

int Naive2DRenderer::run()
{
	mutex.lock(); // Wait for workerReady
	mutex.unlock();

	auto matrix_binder = [](int loc, const void* data) {
		glUniformMatrix4fv(loc, 1, GL_FALSE, (const GLfloat*)data);
	};
	auto vector_binder = [](int loc, const void* data) {
		glUniform4fv(loc, 1, (const GLfloat*)data);
	};
	auto vector3_binder = [](int loc, const void* data) {
		glUniform3fv(loc, 1, (const GLfloat*)data);
	};
	auto float_binder = [](int loc, const void* data) {
		glUniform1fv(loc, 1, (const GLfloat*)data);
	};

	MatrixPointers mats;
	auto std_model_data = [&mats]() -> const void* {
		return mats.model;
	};
	auto std_view_data = [&mats]() -> const void* {
		return mats.view;
	};
	auto std_camera_data  = [&gui]() -> const void* {
		return &gui.getCamera()[0];
	};
	auto std_proj_data = [&mats]() -> const void* {
		return mats.projection;
	};
	auto std_light_data = [&light_position]() -> const void* {
		return &light_position[0];
	};
	auto alpha_data  = [&gui]() -> const void* {
		static const float transparet = 0.5; // Alpha constant goes here
		static const float non_transparet = 1.0;
		if (gui.isTransparent())
			return &transparet;
		else
			return &non_transparet;
	};
	auto red_color_data = []() -> const void* {
		static float color[4] = { 1.0f, 0.0f, 0.0f, 1.0f};
		return color;
	};
	auto yellow_color_data = []() -> const void* {
		static float color[4] = { 1.0f, 1.0f, 0.0f, 1.0f};
		return color;
	};


	const Geo& env = *p_->env;

	RenderDataInput env_pass_input;
	env_pass_input.assign(0, "vertex_position", env.GPUV.data(), env.GPUV.rows(), 3, GL_FLOAT);
	env_pass_input.assign(1, "normal", env.N.data(), env.N.rows(), 3, GL_FLOAT);
	env_pass_input.assign_index(env.F.data(), env.F.rows(), 3);
	RenderPass env_pass(-1,
			env_pass_input,
			{
			  vertex_shader,
			  geometry_shader,
			  fragment_shader
			},
			{ std_model,
			  std_view,
			  std_proj,
			  std_light,
			  std_camera,
			  object_alpha,
			  yellow_diffuse
			  },
			{ "fragment_color" }
			);
	// TODO: Tree Pass

	while (!glfwWindowShouldClose(p_->window)) {
		int window_width, window_height;
		glfwGetFramebufferSize(window, &window_width, &window_height);
		glViewport(0, 0, window_width, window_height);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDepthFunc(GL_LESS);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glCullFace(GL_BACK);

		gui.updateMatrices();
		mats = gui.getMatrixPointers();
		env_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, env.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		// TODO: Render Tree pass
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(p_->window);
	p_->window = nullptr;
	glfwTerminate();

	return 0;
}

