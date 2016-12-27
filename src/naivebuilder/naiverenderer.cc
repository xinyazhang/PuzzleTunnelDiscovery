#include <iostream>
#include "naiverenderer.h"
#include <renderer/quickgl.h>
#include <renderer/render_pass.h>
#include <renderer/gui.h>
#include <omplaux/geo.h>
#include <mutex>
#include <thread>
#include <list>
#include <map>

struct Naive2DRenderer::Private {
	GLFWwindow *window;
	std::unique_ptr<GUI> gui;
	Geo env;
	std::unique_ptr<std::thread> worker;
	std::mutex mutex;

	Private()
	{
		mutex.lock();
	}

	~Private()
	{
		if (worker)
			worker->join();
	}

	struct VF {
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
		Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	};

	VF wireframe, clear_cubes, solid_cubes;
	std::vector<VF> lines;
	std::list<VF> dynlines;
	int next_token = 0;
	std::map<int, decltype(dynlines)::iterator> token_to_iter;

	bool wireframe_dirty = false;
	bool clear_cubes_dirty = false;
	bool solid_cubes_dirty = false;
	bool line_dirty = false;
	bool dynline_dirty = false;
	size_t line_nelements = 0;

	template<typename Container>
	void update_to_render_pass(const Container& line_container, RenderPass& pass)
	{
		size_t new_nelem = line_nelements;
		for (const auto& line : line_container) {
			size_t nelem = line.V.rows();
			new_nelem += nelem;
		}
		pass.updateVBO(0, nullptr, new_nelem); // Allocate the buffer
		line_nelements = 0;
		for (const auto& line : line_container) {
			const auto& V = line.V;
			size_t nelem = V.rows();
			pass.overwriteVBO(0, V.data(), nelem, line_nelements);
			line_nelements += nelem;
		}
	}

	template<typename Container>
	static void render_lines(const Container& lines)
	{
		int first = 0;
		for (const auto& line: lines) {
			CHECK_GL_ERROR(glDrawArrays(GL_LINE_STRIP, first, line.V.rows()));
			first += line.V.rows();
		}
	}

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
		const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& src,
		Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& tgt,
		Scalar base
		)
	{
		int newblkrow = tgt.rows();
		tgt.conservativeResize(tgt.rows() + src.rows(), src.cols());
		tgt.block(newblkrow, 0, src.rows(), tgt.cols()) = src.array() + base;
	}

	static void concatenate(Eigen::MatrixXd& V,
			Eigen::MatrixXi& F,
			VF& vf)
	{
		//std::cerr << "Incoming VF\n" << V <<"\n" << F << std::endl;
		append_matrix<int>(F, vf.F, vf.V.rows());
		append_matrix<float>(V.cast<float>(), vf.V, 0.0f);
#if 0
		std::cerr << "Appended VF\n" << vf.V <<"\n" << vf.F << std::endl;
		std::cerr << "Test V(0,0)\n" << vf.V(0,0) << std::endl;
		std::cerr << "Appended VF sizes\n" << vf.V.rows() <<"\n" << vf.F.rows() << std::endl;
#endif
	}
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

void Naive2DRenderer::addLine(const Eigen::MatrixXd& LV)
{
	std::lock_guard<std::mutex> guard(p_->mutex);
	p_->lines.emplace_back();
	p_->append_matrix<float>(LV.cast<float>(), p_->lines.back().V, 0.0f);
	p_->line_dirty = true;
}

int Naive2DRenderer::addDynamicLine(const Eigen::MatrixXd& LV)
{
	std::lock_guard<std::mutex> guard(p_->mutex);
	auto iter = p_->dynlines.emplace(p_->dynlines.end());
	p_->append_matrix<float>(LV.cast<float>(), p_->dynlines.back().V, 0.0f);
	int token= p_->next_token++;
	p_->token_to_iter[token] = iter;
	p_->dynline_dirty = true;
	return token;
}

void Naive2DRenderer::removeDynamicLine(int token)
{
	std::lock_guard<std::mutex> guard(p_->mutex);
	p_->dynlines.erase(p_->token_to_iter[token]);
	p_->token_to_iter.erase(token);
	p_->dynline_dirty = true;
}

void Naive2DRenderer::init()
{
	p_->window = init_glefw(1440, 900, "Naive Renderer");
	p_->gui.reset(new GUI(p_->window));
}

void Naive2DRenderer::launch_worker(std::function<int(NaiveRenderer*)> fn)
{
	p_->worker.reset(new std::thread(fn, this));
}

void Naive2DRenderer::setEnv(const Geo* env)
{
	p_->env = *env;
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

const char* line_vs =
#include "shaders/line.vert"
;

const char* line_fs =
#include "shaders/line.frag"
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
	p_->gui->setOrthogonalProjection(true);
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
	auto blue_color_data = []() -> const void* {
		static float color[4] = { 0.0f, 0.0f, 1.0f, 1.0f};
		return color;
	};
	auto white_color_data = []() -> const void* {
		static float color[4] = { 1.0f, 1.0f, 1.0f, 1.0f};
		return color;
	};
	auto green_color_data = []() -> const void* {
		static float color[4] = { 0.0f, 1.0f, 0.0f, 1.0f};
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
	ShaderUniform blue_diffuse = { "diffuse", vector_binder, blue_color_data };
	ShaderUniform white_diffuse = { "diffuse", vector_binder, white_color_data };
	ShaderUniform green_diffuse = { "diffuse", vector_binder, green_color_data };

	const Geo& env = p_->env;

	RenderDataInput env_pass_input;
	env_pass_input.assign(0, "vertex_position", env.GPUV.data(), env.GPUV.rows(), 3, GL_FLOAT);
	//env_pass_input.assign(1, "normal", env.N.data(), env.N.rows(), 3, GL_FLOAT);
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
			  blue_diffuse
			  },
			{ "fragment_color" }
			);
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
			  white_diffuse
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
			  red_diffuse
			  },
			{ "fragment_color" }
			);

	RenderDataInput path_pass_input;
	path_pass_input.assign(0, "vertex_position", nullptr, 0, 3, GL_FLOAT);
	RenderPass path_pass(-1,
			path_pass_input,
			{
			  line_vs,
			  nullptr,
			  line_fs
			},
			{ std_model,
			  std_view,
			  std_proj,
			  object_alpha,
			  red_diffuse
			  },
			{ "fragment_color" }
			);

	RenderDataInput dynline_pass_input;
	dynline_pass_input.assign(0, "vertex_position", nullptr, 0, 3, GL_FLOAT);
	RenderPass dynline_pass(-1,
			dynline_pass_input,
			{
			  line_vs,
			  nullptr,
			  line_fs
			},
			{ std_model,
			  std_view,
			  std_proj,
			  object_alpha,
			  green_diffuse
			  },
			{ "fragment_color" }
			);

	//std::cerr << "ENV Faces\n " << env.F << "\nVertices\n" << env.GPUV << std::endl;
	while (!glfwWindowShouldClose(p_->window)) {
		int window_width, window_height;
		glfwGetFramebufferSize(p_->window, &window_width, &window_height);
		glViewport(0, 0, window_width, window_height);
		glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_BLEND);
		glEnable(GL_CULL_FACE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDepthFunc(GL_LESS);
		// glDepthRangef(-1.0, 1.0);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glCullFace(GL_BACK);

		gui->updateMatrices();
		mats = gui->getMatrixPointers();
#if 0
		env_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, env.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
#endif
#if 1
		p_->mutex.lock();

#if 0
		solid_pass.setup();
		if (p_->solid_cubes_dirty) {
			solid_pass.updateVBO(0, p_->solid_cubes.V.data(), p_->solid_cubes.V.rows());
			solid_pass.updateIndex(p_->solid_cubes.F.data(), p_->solid_cubes.F.rows());
			p_->solid_cubes_dirty = false;
		}
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->solid_cubes.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
#endif

#if 1
		wireframe_pass.setup();
		if (p_->wireframe_dirty) {
			wireframe_pass.updateVBO(0, p_->wireframe.V.data(), p_->wireframe.V.rows());
			wireframe_pass.updateIndex(p_->wireframe.F.data(), p_->wireframe.F.rows());
			p_->wireframe_dirty = false;
		}
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->wireframe.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif

#if 0
		clear_pass.setup();
		if (p_->clear_cubes_dirty) {
			clear_pass.updateVBO(0, p_->clear_cubes.V.data(), p_->clear_cubes.V.rows());
			clear_pass.updateIndex(p_->clear_cubes.F.data(), p_->clear_cubes.F.rows());
			p_->clear_cubes_dirty = false;
		}
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, p_->clear_cubes.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
#endif

#if 1 
		path_pass.setup();
		if (p_->line_dirty) {
			p_->update_to_render_pass(p_->lines, path_pass);
			p_->line_dirty = false;
		}
		p_->render_lines(p_->lines);
#endif

		dynline_pass.setup();
		if (p_->dynline_dirty) {
			p_->update_to_render_pass(p_->dynlines, dynline_pass);
			p_->dynline_dirty = false;
		}
		p_->render_lines(p_->dynlines);
#if 0
		std::cerr << "ENV Faces\n " << env.F << "\nVertices\n" << env.V << std::endl;
		//std::cerr << "ENV Faces\n " << env.F.rows() << "\nVertices\n" << env.V.rows() << std::endl;
		std::cerr << " Faces " << p_->wireframe.F.rows()
			  << "\t" << p_->clear_cubes.F.rows()
			  << "\t" << p_->solid_cubes.F.rows()
			  << std::endl;
#endif
		p_->mutex.unlock();
#endif
#if 1
		env_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, env.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
#endif

		glfwPollEvents();
		glfwSwapBuffers(p_->window);
	}
	glfwDestroyWindow(p_->window);
	p_->window = nullptr;
	//glfwTerminate();

	return 0;
}

