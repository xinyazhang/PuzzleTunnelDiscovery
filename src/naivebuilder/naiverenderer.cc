#include <iostream>
#include "naiverenderer.h"
#include <renderer/quickgl.h>
#include <renderer/render_pass.h>
#include <renderer/gui.h>
#include <omplaux/geo.h>
#include <mutex>
#include <thread>

struct Naive2DRenderer::Private {
	GLFWwindow *window;
	std::unique_ptr<GUI> gui;
	const Geo* env;
	std::unique_ptr<std::thread> worker;
	std::mutex mutex;

	Private()
	{
		mutex.lock();
	}

	struct VF {
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
		Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	};

	VF wireframe, clear_cubes, solid_cubes;

	void create_cube(const Eigen::VectorXd& , // Well, it turns out we don't even use center
			const Eigen::VectorXd& mins,
			const Eigen::VectorXd& maxs,
			Eigen::MatrixXd& cubev,
			Eigen::MatrixXi& cubei)
	{
		// FIXME: dup code from boolean/main.cc
		double xmin = mins(0), ymin = mins(1), zmin = -0.5;
		double xmax = maxs(0), ymax = maxs(1), zmax =  0.5;
		cubev.resize(8, 3);
		cubev.row(0) << xmin, ymin, zmin;
		cubev.row(1) << xmax, ymin, zmin;
		cubev.row(2) << xmax, ymax, zmin;
		cubev.row(3) << xmin, ymax, zmin;
		cubev.row(4) << xmin, ymin, zmax;
		cubev.row(5) << xmax, ymin, zmax;
		cubev.row(6) << xmax, ymax, zmax;
		cubev.row(7) << xmin, ymax, zmax;
		cubei.resize(12,3);
		cubei.row(0) << 3,2,1;
		cubei.row(1) << 1,0,3;
		cubei.row(2) << 1,2,5;
		cubei.row(3) << 2,6,5;
		cubei.row(4) << 4,5,7;
		cubei.row(5) << 5,6,7;
		cubei.row(6) << 0,1,4;
		cubei.row(7) << 1,5,4;
		cubei.row(8) << 2,3,7;
		cubei.row(9) << 2,7,6;
		cubei.row(10)<< 0,4,7;
		cubei.row(11)<< 0,7,3;
	}

	template <typename Scalar>
	static void append_matrix(
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& src,
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& tgt,
		Scalar base
		)
	{
		int newblkrow = tgt.rows();
		tgt.conservativeResize(tgt.rows() + src.rows(), tgt.cols());
		tgt.block(newblkrow, 0, src.rows(), tgt.cols()) = src.array() + base;
	}

	static void concatenate(Eigen::MatrixXd& V,
			Eigen::MatrixXi& F,
			VF& vf)
	{
		append_matrix<double>(V, vf.V, 0.0);
		append_matrix<int>(F, vf.F, vf.F.rows());
	}

	bool wireframe_dirty = false;
	bool clear_cubes_dirty = false;
	bool solid_cubes_dirty = false;
};

Naive2DRenderer::Naive2DRenderer()
	:p_(new Naive2DRenderer::Private)
{
}

Naive2DRenderer::~Naive2DRenderer()
{
}

void Naive2DRenderer::addSplit(const Eigen::VectorXd& center, 
		      const Eigen::VectorXd& mins,
		      const Eigen::VectorXd& maxs)
{
	std::lock_guard<std::mutex> guard(p_->mutex);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	p_->create_cube(center, mins, maxs, V, F);
	p_->concatenate(V, F, p_->wireframe);
	p_->wireframe_dirty = true;
}

void Naive2DRenderer::addCertain(const Eigen::VectorXd& center, 
			const Eigen::VectorXd& mins,
			const Eigen::VectorXd& maxs,
			bool isfree)
{
	std::lock_guard<std::mutex> guard(p_->mutex);
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	p_->create_cube(center, mins, maxs, V, F);
	if (isfree) {
		p_->concatenate(V, F, p_->clear_cubes);
		p_->clear_cubes_dirty = true;
	} else {
		p_->concatenate(V, F, p_->solid_cubes);
		p_->solid_cubes_dirty = true;
	}
}

void Naive2DRenderer::init()
{
	p_->window = init_glefw(800, 600, "Naive Renderer");
	p_->gui.reset(new GUI(p_->window));
}

void Naive2DRenderer::launch_worker(std::function<int(NaiveRenderer*)> fn)
{
	p_->worker.reset(new std::thread(fn, this));
}

void Naive2DRenderer::setEnv(const Geo* env)
{
	p_->env = env;
}

void Naive2DRenderer::workerReady()
{
	p_->mutex.unlock();
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
	p_->mutex.lock(); // Wait for workerReady
	p_->mutex.unlock();

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
	glm::vec4 light_position = glm::vec4(0.0f, 0.0f, 30.0f, 1.0f);
	auto std_model_data = [&mats]() -> const void* {
		return mats.model;
	};
	auto std_view_data = [&mats]() -> const void* {
		return mats.view;
	};
	GUI* gui = p_->gui.get();
	auto std_camera_data  = [gui]() -> const void* {
		return &gui->getCamera()[0];
	};
	auto std_proj_data = [&mats]() -> const void* {
		return mats.projection;
	};
	auto std_light_data = [&light_position]() -> const void* {
		return &light_position[0];
	};
	auto alpha_data  = [gui]() -> const void* {
		static const float transparet = 0.5; // Alpha constant goes here
		static const float non_transparet = 1.0;
		if (gui->isTransparent())
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

	ShaderUniform std_model = { "model", matrix_binder, std_model_data };
	ShaderUniform std_view = { "view", matrix_binder, std_view_data };
	ShaderUniform std_camera = { "camera_position", vector3_binder, std_camera_data };
	ShaderUniform std_proj = { "projection", matrix_binder, std_proj_data };
	ShaderUniform std_light = { "light_position", vector_binder, std_light_data };
	ShaderUniform object_alpha = { "alpha", float_binder, alpha_data };
	ShaderUniform red_diffuse = { "diffuse", vector_binder, red_color_data };
	ShaderUniform yellow_diffuse = { "diffuse", vector_binder, yellow_color_data };

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
	RenderDataInput wireframe_pass_input;
	wireframe_pass_input.assign(0, "vertex_position", nullptr, 0, 3, GL_FLOAT);
	wireframe_pass_input.assign_index(nullptr, 0, 3);
	RenderPass wireframe_pass(-1,
			wireframe_pass_input,
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

	RenderDataInput clear_pass_input;
	clear_pass_input.assign(0, "vertex_position", nullptr, 0, 3, GL_FLOAT);
	clear_pass_input.assign_index(nullptr, 0, 3);
	RenderPass clear_pass(-1,
			clear_pass_input,
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
	RenderDataInput solid_pass_input;
	solid_pass_input.assign(0, "vertex_position", nullptr, 0, 3, GL_FLOAT);
	solid_pass_input.assign_index(nullptr, 0, 3);
	RenderPass solid_pass(-1,
			solid_pass_input,
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

	while (!glfwWindowShouldClose(p_->window)) {
		int window_width, window_height;
		glfwGetFramebufferSize(p_->window, &window_width, &window_height);
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

		gui->updateMatrices();
		mats = gui->getMatrixPointers();
		env_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, env.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		p_->mutex.lock();

		if (p_->wireframe_dirty) {
			wireframe_pass.updateVBO(0, p_->wireframe.V.data(), p_->wireframe.V.rows());
			wireframe_pass.updateIndex(p_->wireframe.F.data(), p_->wireframe.F.rows());
			p_->wireframe_dirty = false;
		}
		wireframe_pass.setup();
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->wireframe.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		if (p_->clear_cubes_dirty) {
			clear_pass.updateVBO(0, p_->clear_cubes.V.data(), p_->clear_cubes.V.rows());
			clear_pass.updateIndex(p_->clear_cubes.F.data(), p_->clear_cubes.F.rows());
			p_->clear_cubes_dirty = false;
		}
		clear_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->clear_cubes.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));

		if (p_->solid_cubes_dirty) {
			solid_pass.updateVBO(0, p_->solid_cubes.V.data(), p_->solid_cubes.V.rows());
			solid_pass.updateIndex(p_->solid_cubes.F.data(), p_->solid_cubes.F.rows());
			p_->solid_cubes_dirty = false;
		}
		solid_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->solid_cubes.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));

		p_->mutex.unlock();

		glfwPollEvents();
		glfwSwapBuffers(p_->window);
	}
	glfwDestroyWindow(p_->window);
	p_->window = nullptr;
	glfwTerminate();

	return 0;
}

