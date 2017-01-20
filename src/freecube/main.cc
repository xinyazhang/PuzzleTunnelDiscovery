#include <renderer/quickgl.h>
#include <renderer/render_pass.h>
#include <renderer/gui.h>
#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include <Eigen/StdVector>
#include <Eigen/Geometry> 
//#include <igl/barycenter.h>

#define SANITY_CHECK 0

using std::string;

const char* vertex_shader =
#include "shaders/default.vert"
;

const char* geometry_shader =
#include "shaders/default.geom"
;

const char* fragment_shader =
#include "shaders/default.frag"
;

const char* floor_fragment_shader =
#include "shaders/floor.frag"
;

int main(int argc, char* argv[])
{
	GLFWwindow *window = init_glefw(800, 600, "Alpha animation");
	GUI gui(window);

#if 0
	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
	string envcvxpn;
	gui.setCameraDistance(150.0f);
#elif 0
	string robotfn = "../res/simple/robot.obj";
	string envfn = "../res/simple/FullTorus.obj";
	string envcvxpn = "../res/simple/cvx/FullTorus";
	string pathfn = "1.path";
#elif 1
	string robotfn = "../res/simple/LongStick.obj";
	string envfn = "../res/simple/mFixedElkMeetsCube.obj";
	string pathfn = "agg.path";
	string envcvxpn;
#elif 1
	string robotfn = "../res/simple/mediumstick.obj";
	string envfn = "../res/simple/boxwithhole2.obj";
	string pathfn = "../res/simple/boxreference.path";
	string envcvxpn = "../res/simple/cvx/boxwithhole";
#else
	string robotfn = "../res/alpha/rob-1.2.obj";
	string envfn = "../res/alpha/env-1.2.obj";
	string envcvxpn = "../res/alpha/cvx/env-1.2";
	string pathfn = "1.path";
	gui.setCameraDistance(150.0f);
#endif
	Geo robot, env;
	Path path;
	robot.read(robotfn);
	env.read(envfn);
	if (!envcvxpn.empty())
		env.readcvx(envcvxpn);
	path.readPath(pathfn);
#if 0
	robot.center << 16.973146438598633, 1.2278236150741577, 10.204807281494141; // From OMPL.app, no idea how they get this.
#else
	robot.center << 0.0, 0.0, 0.0;
#endif
	//robot.center << 17.491058349609375, 1.386110782623291, 10.115392684936523; // It changed, we also don't know why, but this one doesn't work.

	double t = 0.0;
	Path::GLMatrixd robot_transform_matrix = path.interpolate(robot, 0.0);
	std::cerr << "Initial transformation matrix" << std::endl << robot_transform_matrix << std::endl;
	Path::GLMatrix alpha_model_matrix = robot_transform_matrix.cast<float>();
	glm::vec4 light_position = glm::vec4(0.0f, 100.0f, 0.0f, 1.0f);
	MatrixPointers mats;
	ClearanceCalculator<fcl::OBBRSS<double>> cc(robot, env);
	cc.setC(-100,100);
	
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
	auto robot_model_data = [&alpha_model_matrix]() -> const void* {
		return alpha_model_matrix.data();
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
	ShaderUniform robot_model = { "model", matrix_binder, robot_model_data };

	std::cerr << robot.GPUV << std::endl;
	RenderDataInput robot_pass_input;
	robot_pass_input.assign(0, "vertex_position", robot.GPUV.data(), robot.GPUV.rows(), 3, GL_FLOAT);
	robot_pass_input.assign(1, "normal", robot.N.data(), robot.N.rows(), 3, GL_FLOAT);
	robot_pass_input.assign_index(robot.F.data(), robot.F.rows(), 3);
	RenderPass robot_pass(-1,
			robot_pass_input,
			{
			  vertex_shader,
			  geometry_shader,
			  fragment_shader
			},
			{ robot_model,
			  std_view,
			  std_proj,
			  std_light,
			  std_camera,
			  object_alpha,
			  red_diffuse 
			  },
			{ "fragment_color" }
			);

	RenderDataInput obs_pass_input;
	obs_pass_input.assign(0, "vertex_position", env.GPUV.data(), env.GPUV.rows(), 3, GL_FLOAT);
	obs_pass_input.assign(1, "normal", env.N.data(), env.N.rows(), 3, GL_FLOAT);
	obs_pass_input.assign_index(env.F.data(), env.F.rows(), 3);
	RenderPass obs_pass(-1,
			obs_pass_input,
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

	std::cerr.precision(17);
	while (!glfwWindowShouldClose(window)) {
		// Setup some basic window stuff.
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

		t += 1.0/30.0;
		auto state = path.interpolateState(t);
		// state << -0.065337, -0.624994, 8.32952,  -2.47891, 1.04311, -1.05538;
		// state << 0.109556, -0.799887, 7.45506, -2.3654, 1.07839, -0.954136;
#if 0
		robot_transform_matrix = path.interpolate(robot, t);
		//robot_transform_matrix(0, 3) = t;
		alpha_model_matrix = robot_transform_matrix.cast<float>();
		// std::cerr << "Transform matrix:\n " << robot_transform_matrix << std::endl;
#else
		robot_transform_matrix = Path::stateToMatrix<double>(state, robot.center);
		alpha_model_matrix = robot_transform_matrix.cast<float>();
#endif

		robot_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, robot.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		obs_pass.setup();
		CHECK_GL_ERROR(glDrawElements(GL_TRIANGLES, env.F.rows() * 3,
					GL_UNSIGNED_INT,
					0));
		//std::cerr << t << std::endl;
		// Poll and swap.
		glfwPollEvents();
		glfwSwapBuffers(window);

		bool isfree;
		double pd;
		(void)cc.getCertainCube(state, isfree, &pd);
		std::cerr << "Free: " << (isfree ? "true" : "false") << std::endl;
		std::cerr << "D: " << pd << std::endl;
#if 0
		double mindist = cc.getDistance(robot_transform_matrix);
		auto clearance = cc.getClearanceCube(robot_transform_matrix, mindist);
		std::cerr << "Distance " << mindist
		          << "\tClearance: " << clearance
			  << std::endl;
#endif
#if SANITY_CHECK
		auto san = cc.sanityCheck(robot_transform_matrix, clearance);
		std::cerr << "\tSanity: " << san << std::endl;
#endif
	} glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
