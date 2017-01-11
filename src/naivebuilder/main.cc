#include "naiverenderer.h"
#include "naiveclearance.h"
#include "naivespace.h"
#include "vis2d.h"
#define ENABLE_DFS 0
#include <goct/goctree.h>
#include <goct/gbuilder.h>
#include <string>

using std::string;

int worker(NaiveRenderer* renderer)
{
	string envfn = "../res/simple/Torus.obj";
	// string envfn = "../res/simple/FullTorus.obj";
	Geo env;
	env.read(envfn);
	env.V.block(0, 2, env.V.rows(), 1) *= 0.0001;
	env.GPUV.block(0, 2, env.V.rows(), 1) *= 0.001f;

	renderer->setEnv(&env);
	NaiveClearance cc(env);

	NaiveVisualizer::setRenderer(renderer);
	using Builder = GOctreePathBuilder<2,
	      double,
	      decltype(cc),
	      TranslationOnlySpace<2,double>,
	      //NullVisualizer>;
	      NaiveVisualizer>;
	using Coord = typename Builder::Coord;
	Coord min, max, res;

	double tmin = -10;
	double tmax = 10;
	min << tmin, tmin;
	max << tmax, tmax;
	cc.setC(tmin, tmax);

	res = (max - min) / 20000.0; // FIXME: how to calculate a resolution?

	Builder builder;
	builder.setupSpace(min, max, res);
	Coord init_p, goal_p;
	init_p << -1.0, -1.0;
	goal_p << -9.0, 0.0;

	builder.setupInit(init_p);
	builder.setupGoal(goal_p);

	renderer->workerReady();

	builder.buildOcTree(cc);
	auto path = builder.buildPath();
	if (!path.empty()) {
		std::cerr << path << std::endl;
		Eigen::MatrixXd np;
		np.resize(path.size(), path.front().size() + 1); // Note: 2D only
		for (size_t i = 0; i < path.size(); i++) {
			np.row(i) = path[i];
			np(i, 2) = 2.0; // Note: 2D only
		}
		renderer->addLine(np);
		std::cerr << "Done\n";
	}
	Builder::VIS::pause();
	std::cerr << "Worker thread exited\n";

	return 0;
}

int main(int argc, char* argv[])
{
	Naive2DRenderer render;
	render.init();
	render.launch_worker(worker);
	return render.run();
}
