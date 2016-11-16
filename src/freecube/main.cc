#include "quickgl.h"
#include <iostream>
#include <fstream>
#include <string>
#include <fcl/fcl.h> // This incldued eigen as well.
#include <Eigen/StdVector>
#include <Eigen/Geometry> 
#include <igl/readPLY.h>
//#include <igl/barycenter.h>

using std::string;

struct Geo {
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> F;

	Eigen::Vector3d center;

	void readPLY(const string& fn)
	{
		igl::readPLY(fn, V, F);
		center << 0.0, 0.0, 0.0; // Origin
	}
};

struct Path {
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
			fin >> qx >> qy >> qz >> qw;
			Q.emplace_back(qw, qx, qy, qz);
		}
		for (size_t i = 0; i < T.size(); i++) {
			std::cerr << T[i].transpose() << "\t" << Q[i].vec().transpose() << " " << Q[i].w() << std::endl;
		}
	}
};

#if 0
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
#endif

int main(int argc, char* argv[])
{
#if 0
	GLFWwindow *window = init_glefw();
	GUI gui(window);
#endif

	string robotfn = "../res/alpha/robot.ply";
	string envfn = "../res/alpha/env-1.1.ply";
	string pathfn = "../res/alpha/alpha-1.1.path";
	Geo robot, env;
	Path path;
	robot.readPLY(robotfn);
	env.readPLY(envfn);
	path.readPath(pathfn);
	
	return 0;
}
