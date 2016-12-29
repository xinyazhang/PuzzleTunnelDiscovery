#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include <omplaux/scene_bounding_box.h>
#include <vecio/arrayvecio.h>
#include <goct/goctree.h>
#include <goct/gbuilder.h>
#include "space.h"
#include "textvisualizer.h"

using std::string;

int main(int argc, char* argv[])
{
#if 0
	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
#else
	string robotfn = "../res/simple/robot.obj";
	string envfn = "../res/simple/env2.obj";
	string pathfn = "../res/simple/naive2.path";
#endif
	Geo robot, env;
	Path path;

	robot.read(robotfn);
	env.read(envfn);
	path.readPath(pathfn);
#if 0
	robot.center << 16.973146438598633, 1.2278236150741577, 10.204807281494141; // From OMPL.app, no idea how they get this.
#else
	robot.center << 0.0, 0.0, 0.0;
#endif

	ClearanceCalculator<fcl::OBBRSS<double>> cc(robot, env);

	typename GOcTreeNode<6, double>::Coord min, max, res;
#if 0
	min << -100.0, -100.0, -100.0, -M_PI/2.0,      0.0,      0.0;
	max <<  100.0,  100.0,  100.0,  M_PI/2.0, M_PI * 2, M_PI * 2;
	cc.setC(-100,100);
#elif 0
	min << -10.0, -10.0, -10.0, -M_PI/2.0,      0.0,      0.0;
	max <<  10.0,  10.0,  10.0,  M_PI/2.0, M_PI * 2, M_PI * 2;
	cc.setC(-10, 10);
#elif 0
	double tmin = -30;
	double tmax = 30;
	min << tmin, tmin, tmin, M_PI*0.0, -M_PI/2.0, M_PI*0.0;
	max << tmax, tmax, tmax, M_PI*2.0,  M_PI/2.0, M_PI*2.0;
	cc.setC(tmin, tmax);
#else
	double bbmin, bbmax;
	omplaux::calculateSceneBoundingBox(robot, env, path, bbmin, bbmax);
	min << bbmin, bbmin, bbmin, -M_PI, -M_PI/2.0, -M_PI;
	max << bbmax, bbmax, bbmax,  M_PI,  M_PI/2.0,  M_PI;
	std::cerr << "Bounding box\n"
	          << "\tmin: " << min.transpose() << "\n"
		  << "\tmax: " << max.transpose() << std::endl;
	cc.setC(bbmin, bbmax);
#endif
	res = (max - min) / 20000.0; // FIXME: how to calculate a resolution?

	using Builder = GOctreePathBuilder<6,
	      double,
	      decltype(cc),
	      TranslationWithEulerAngleGroup<6, double>,
	      //NullVisualizer
	      TextVisualizer
	      >;
	Builder builder;
	builder.setupSpace(min, max, res);
	double init_t = 0.0;
	//double init_t = path.T.size() - 2;
	builder.setupInit(Path::matrixToState(path.interpolate(robot, init_t)));
	//double end_t = path.T.size() - 1;
	double end_t = 1.0;
	builder.setupGoal(Path::matrixToState(path.interpolate(robot, end_t)));
	builder.buildOcTree(cc);
	std::cout << builder.buildPath();
	Builder::VIS::pause();

	return 0;
}
