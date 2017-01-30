#define OMPL_CC_DISCRETE_PD 0
#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include <omplaux/scene_bounding_box.h>
#include <vecio/arrayvecio.h>
#include <goct/goctree.h>
// #define ENABLE_DFS 1
// #define PRIORITIZE_SHORTEST_PATH 1
// #define ENABLE_DFS 0
#define PRIORITIZE_SHORTEST_PATH 1
#define PRIORITIZE_CLEARER_CUBE 1
#define ENABLE_DIJKSTRA 1
#include <goct/gbuilder.h>
#include "space.h"
#include "textvisualizer.h"
#include "vis6d.h"

using std::string;

int worker(NaiveRenderer* renderer)
{
	string known_path;
	Eigen::Vector3d robotcenter { Eigen::Vector3d::Zero() };
	string envcvxpn;
	string tetprefix;
#if 0
	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
#elif 0
	string robotfn = "../res/simple/robot.obj";
	string envfn = "../res/simple/FullTorus.obj";
	string pathfn = "../res/simple/naive2.path";
	envcvxpn = "../res/simple/cvx/FullTorus";
#elif 0
	string robotfn = "../res/simple/mediumstick.obj";
	string envfn = "../res/simple/FullTorus.obj";
	string pathfn = "../res/simple/sticktorus.path";
	envcvxpn = "../res/simple/cvx/FullTorus";
#elif 0
	string robotfn = "../res/simple/robot.obj";
	// string robotfn = "../res/simple/LongStick.obj";
	string envfn = "../res/simple/mFixedElkMeetsCube.obj";
	string pathfn = "../res/simple/naiveelk.path";
	envcvxpn = "../res/simple/cvx/ElkMeetsCube";
#elif 1
	string robotfn = "../res/simple/mediumstick.obj";
	string envfn = "../res/simple/boxwithhole2.obj";
	string pathfn = "../res/simple/box.path";
	tetprefix = "../res/simple/tet/boxwithhole2.1";
	// envcvxpn = "../res/simple/cvx/boxrefine/boxwithhole";
	known_path = "../res/simple/boxreference.path";
	// robotcenter << 0.0, 0.0, 0.0;
	robotcenter << -1.1920928955078125e-07, 0.0, 2.384185791015625e-07;
#else
	string robotfn = "../res/alpha/rob-1.2.obj";
	string envfn = "../res/alpha/env-1.2.obj";
	string pathfn = "../res/alpha/ver1.2.path";
	envcvxpn = "../res/alpha/cvx/env-1.2";
	robotcenter << 16.973146438598633, 1.2278236150741577, 10.204807281494141;
#endif
	Geo robot, env;
	Path path;

	robot.read(robotfn);
	env.read(envfn);
	if (!envcvxpn.empty())
		env.readcvx(envcvxpn);
	if (!tetprefix.empty())
		env.readtet(tetprefix);
	path.readPath(pathfn);
	robot.center = robotcenter;
#if 0
#if 1
	robot.center << 16.973146438598633, 1.2278236150741577, 10.204807281494141; // From OMPL.app, no idea how they get this.
#else
	robot.center << 0.0, 0.0, 0.0;
#endif
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
	// double dalpha = M_PI/128.0;
	double dalpha = M_PI;
	// min << bbmin, bbmin, bbmin, -M_PI, -M_PI/2.0, -M_PI;
	// max << bbmax, bbmax, bbmax,  M_PI,  M_PI/2.0,  M_PI;
	min << bbmin, bbmin, bbmin, -dalpha, -dalpha/2.0, -dalpha;
	max << bbmax, bbmax, bbmax,  dalpha,  dalpha/2.0,  dalpha;
	std::cerr << "Bounding box\n"
	          << "\tmin: " << min.transpose() << "\n"
		  << "\tmax: " << max.transpose() << std::endl;
	cc.setC(bbmin, bbmax);
	// cc.setC(-100, 100);
	cc.setDAlpha(dalpha);
#endif
	res = (max - min) / 20000.0; // FIXME: how to calculate a resolution?

	using Builder = GOctreePathBuilder<6,
	      double,
	      decltype(cc),
	      TranslationWithEulerAngleGroup<6, double>,
	      // NullVisualizer
	      TextVisualizer
	      // NaiveVisualizer6D
	      >;
	Builder::VIS::setRenderer(renderer);
	Builder builder;
	builder.setupSpace(min, max, res);
	// double init_t = 0.0;
	// double init_t = path.T.size() - 2;
	// builder.setupInit(Path::matrixToState(path.interpolate(robot, init_t)));
	builder.setupInit(path.pathToState(0));
	//double end_t = path.T.size() - 1;
	// double end_t = 1.0;
	// builder.setupGoal(Path::matrixToState(path.interpolate(robot, end_t)));
	builder.setupGoal(path.pathToState(1));
	renderer->workerReady();

	using Clock = std::chrono::high_resolution_clock;
	auto t1 = Clock::now();
	if (!known_path.empty()) {
		std::cerr.precision(17);
		Path pathsln;
		pathsln.readPath(known_path);
		size_t fps = 15;
		builder.init_builder(cc);
		// for (size_t i = 0; i < (pathsln.T.size() - 1) * fps; i++) {
		// for (size_t i = 40 * fps; i < 41 * fps; i++) {
		// size_t i = 38.8 * fps; {
		// size_t i = 41.6 * fps; {
		size_t i = 40 * fps; {
			double t = double(i) / fps;
			auto state = pathsln.interpolateState(t);
			
			std::cerr << "Time: " << t << "\tState: " << state.transpose() << std::endl;
			auto node = builder.determinize_cube(state);
			std::cerr << "\tSAN check: " << *node << std::endl;
			bool isfree;
			(void)cc.getCertainCube(state, isfree);
			std::cerr << "\tState Free: " << (isfree ? "true" : "false") << std::endl;
			double pd = -1.0;
			auto d = cc.getCertainCube(node->getMedian(), isfree, &pd);
			std::cerr << "\tMedian Free: " << (isfree ? "true" : "false") << std::endl;
			std::cerr << "\tMedian Certain: " << d.transpose() << std::endl;
			std::cerr << "\tPD^T: " << pd << std::endl;
		}
	} else {
		builder.buildOcTree(cc);
	}
	auto t2 = Clock::now();
	std::chrono::duration<double, std::milli> dur = t2 - t1;

	auto slnpath = builder.buildPath();
	std::cout << "Solution path: " << std::endl;
	for (const auto& state: slnpath)
		std::cout << Path::stateToPath<double>(state.block<6, 1>(0,0)).transpose() << std::endl;
	Builder::VIS::pause();
	std::cerr << "Configuration: ENABLE_DFS = " << ENABLE_DFS
	          << "\tPRIORITIZE_SHORTEST_PATH = " << PRIORITIZE_SHORTEST_PATH
	          << std::endl;
	std::cerr << "Planning takes " << dur.count() << " ms to complete\n";
	std::cerr << "Maximum cube depth " << builder.getDeepestLevel() << "\n";

	return 0;
}

int main(int argc, char* argv[])
{
	Naive3DRenderer render;
	render.init();
	render.launch_worker(worker);
	return render.run();
}
