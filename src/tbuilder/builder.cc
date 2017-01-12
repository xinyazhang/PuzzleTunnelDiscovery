#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include <omplaux/scene_bounding_box.h>
#include <vecio/arrayvecio.h>
#include <goct/goctree.h>
#define ENABLE_DFS 0
#define PRIORITIZE_SHORTEST_PATH 1
#define PRIORITIZE_CLEARER_CUBE 1
#define ENABLE_DIJKSTRA 1
#include <goct/gbuilder.h>
#include "naivespace.h"
#include "vis3d.h"
#include "naiverenderer.h"
#include "clearancer.h"
#include <chrono>

using std::string;

int worker(NaiveRenderer* renderer)
{
#if 0
	string robotfn = "../res/simple/robot.obj";
	string envfn = "../res/simple/FullTorus.obj";
	string pathfn = "../res/simple/naive2.path";
#else
	// string robotfn = "../res/simple/robot.obj";
	string robotfn = "../res/simple/LongStick.obj";
	string envfn = "../res/simple/mFixedElkMeetsCube.obj";
	string pathfn = "../res/simple/naiveelk.path";
	string envcvxpn = "../res/simple/cvx/ElkMeetsCube";
#endif
	Geo robot, env; Path path;

	std::cerr << "Robot Reading\n";
	robot.read(robotfn);
	std::cerr << "Robot Read\n";
	std::cerr << "ENV Reading\n";
	env.read(envfn);
	env.readcvx(envcvxpn);
	std::cerr << "ENV Read\n";
	std::cerr << "Path Reading\n";
	path.readPath(pathfn);
	std::cerr << "Path Read\n";
	robot.center << 0.0, 0.0, 0.0;
	std::cerr << "Renderer reading Env\n";
	renderer->setEnv(&env);
	std::cerr << "Renderer read Env\n";

	std::cerr << "CC constructing\n";
	TranslationOnlyClearance<fcl::OBBRSS<double>> cc(robot, env);
	std::cerr << "CC constructed\n";

	typename GOcTreeNode<3, double>::Coord min, max, res;
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
	std::cerr << "Calclating BB\n";
	omplaux::calculateSceneBoundingBox(robot, env, path, bbmin, bbmax);
	min << bbmin, bbmin, bbmin, -M_PI, -M_PI/2.0, -M_PI;
	max << bbmax, bbmax, bbmax,  M_PI,  M_PI/2.0,  M_PI;
	std::cerr << "Bounding box\n"
	          << "\tmin: " << min.transpose() << "\n"
		  << "\tmax: " << max.transpose() << std::endl;
	cc.setC(bbmin, bbmax);
#endif
	res = (max - min) / 1280000.0; // FIXME: how to calculate a resolution?

	using Builder = GOctreePathBuilder<3,
	      double,
	      decltype(cc),
	      TranslationOnlySpace<3, double>,
	      // NullVisualizer
	      NodeCounterVisualizer
	      // NaiveVisualizer3D
	      >;
	Builder::VIS::setRenderer(renderer);
	Builder builder;
	builder.setupSpace(min, max, res);
	double init_t = 0.0;
	//double init_t = path.T.size() - 2;
	auto init_state = Path::matrixToState(path.interpolate(robot, init_t));
	builder.setupInit(init_state.block<3,1>(0,0));
	//double end_t = path.T.size() - 1;
	double end_t = 1.0;
	auto goal_state = Path::matrixToState(path.interpolate(robot, end_t));
	builder.setupGoal(goal_state.block<3,1>(0,0));

	renderer->workerReady();
	std::cerr << "Worker ready\n";

	using Clock = std::chrono::high_resolution_clock;
	auto t1 = Clock::now();
	builder.buildOcTree(cc);
	auto t2 = Clock::now();
	std::chrono::duration<double, std::milli> dur = t2 - t1;
	{
		auto path = builder.buildPath();
		if (!path.empty()) {
			std::cerr << "PATH FOUND:\n" << path << std::endl;
			Eigen::MatrixXd np;
			np.resize(path.size(), path.front().size());
			for (size_t i = 0; i < path.size(); i++) {
				np.row(i) = path[i];
			}
			renderer->addLine(np);
			std::cerr << "Done\n";
		}
	}
	Builder::VIS::pause();
	std::cerr << "Configuration: ENABLE_DFS = " << ENABLE_DFS
	          << "\tPRIORITIZE_SHORTEST_PATH = " << PRIORITIZE_SHORTEST_PATH
	          << std::endl;
	std::cerr << "Planning takes " << dur.count() << " ms to complete\n";
	std::cerr << "Maximum cube depth " << builder.getDeepestLevel() << "\n";
#if COUNT_NODES
	Builder::VIS::showHistogram();
#endif
	auto profiler = cc.getProfiler();
	std::cerr << "FCL statistics\n\t Distance: " << profiler.getAverageClockMs(decltype(profiler)::DISTANCE)
		  << "\n\t Collision: " <<  profiler.getAverageClockMs(decltype(profiler)::COLLISION)
		  << std::endl;
	std::cerr << "Worker thread exited\n";

	return 0;
}

int main(int argc, char* argv[])
{
	Naive3DRenderer render;
	render.init();
	render.launch_worker(worker);
	return render.run();
}
