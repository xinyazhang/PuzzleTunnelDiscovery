#include "quickgl.h"
#include "render_pass.h"
#include "gui.h"
#include "debuggl.h"
#include <iostream>
#include <fstream>
#include <string>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <Eigen/StdVector>
#include <Eigen/Geometry> 
#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>
//#include <igl/barycenter.h>

using std::string;

struct Geo {
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> N;

	Eigen::Vector3d center;

	void read(const string& fn)
	{
		igl::readOBJ(fn, V, F);
		//center << 0.0, 0.0, 0.0; // Origin
		center = V.colwise().mean().cast<double>();
		//center << 16.973146438598633, 1.2278236150741577, 10.204807281494141;
		// From OMPL.app, no idea how they get this.
		igl::per_vertex_normals(V, F, N);
		std::cerr << "center: " << center << std::endl;
#if 0
		std::cerr << N << std::endl;;
#endif
	}
};

struct Path {
	typedef Eigen::Matrix<float, 4, 4, Eigen::ColMajor> GLMatrix;
	typedef Eigen::Matrix<double, 4, 4, Eigen::ColMajor> GLMatrixd;
	Eigen::aligned_vector<Eigen::Vector3d> T;
	Eigen::aligned_vector<Eigen::Quaternion<double>> Q;
	//Eigen::aligned_vector<fcl::Transform3d> M;

	void readPath(const string& fn)
	{
		std::ifstream fin(fn);
		while (true) {
			double x, y, z;
			fin >> x >> y >> z;
			if (fin.eof())
				break;
			T.emplace_back(x, y, z);
			double qx, qy, qz, qw;
			// If no rotation is represented with (0,0,0,1)
			// we know it's xyzw sequence because w = cos(alpha/2) = 1 when
			// alpha = 0.
			fin >> qx >> qy >> qz >> qw;
			Q.emplace_back(qw, qx, qy, qz);
		}
#if 0
		for (size_t i = 0; i < T.size(); i++) {
			std::cerr << T[i].transpose() << "\t" << Q[i].vec().transpose() << " " << Q[i].w() << std::endl;
		}
#endif
		std::cerr << "T size: " << T.size() << std::endl;
	}

	GLMatrix interpolate(const Geo& robot, double t)
	{
		int i = std::floor(t);
		double c = t - double(i);
		int from = i % T.size();
		int to = (i + 1) % T.size();

		GLMatrixd ret;
		ret.setIdentity();
		// Translate to origin
		ret.block<3,1>(0, 3) = -robot.center;

		Eigen::Quaternion<double> Qfrom = Q[from];
		Eigen::Quaternion<double> Qinterp = Qfrom.slerp(c, Q[to]);
		auto rotmat = Qinterp.toRotationMatrix();
		ret.block<3,3>(0,0) = rotmat;
		ret.block<3,1>(0,3) = rotmat * (-robot.center);
		// Translation
		Eigen::Vector3d translate = T[from] * (1 - c) + T[to] * c;
		GLMatrixd trback;
		trback.setIdentity();
		trback.block<3,1>(0, 3) = translate;
		ret = trback * ret; // Trback * Rot * Tr2Origin
		return ret.cast<float>();
	}
};

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
	GLFWwindow *window = init_glefw();
	GUI gui(window);

	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
	Geo robot, env;
	//robot.center << 16.973146438598633, 1.2278236150741577, 10.204807281494141; // From OMPL.app, no idea how they get this.
	Path path;
	robot.read(robotfn);
	env.read(envfn);
	path.readPath(pathfn);

	double t = 0.0;
	Path::GLMatrix alpha_model_matrix = path.interpolate(robot, 0.0);
	glm::vec4 light_position = glm::vec4(0.0f, 100.0f, 0.0f, 1.0f);
	MatrixPointers mats;
	
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

	RenderDataInput robot_pass_input;
	robot_pass_input.assign(0, "vertex_position", robot.V.data(), robot.V.rows(), 3, GL_FLOAT);
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
	obs_pass_input.assign(0, "vertex_position", env.V.data(), env.V.rows(), 3, GL_FLOAT);
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

		t += 1.0/24.0;
		alpha_model_matrix = path.interpolate(robot, t);

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
	}
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
