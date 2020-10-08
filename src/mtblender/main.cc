/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
/*
 * MT Blender -- A program blends the roadMap and Tree
 * 
 * In training the reinforcement learning network, we need the roadmap to
 * figure out global values like the value function and policy function. Hence
 * PRM is required
 * 
 * However, PRM usually failed to find a solution, and to fix this we use RRT,
 * or (our homebrewed) ReRRT to ensure the roadmap are connected.
 */

#include <iostream>
#include <fstream>
#include "config.h"
#include "graph.h"

namespace {
using namespace std;

void usage()
{
	cout << "Usage: mtblender [PRM file] [RRT path file]" << endl;
	cout << "       PRM file comes from ompl.app/demos/SE3RigidBodyPlanning/config_planner.cc:printPlan()" << endl;
	cout << "       RRT path comes from stdout of ompl.app/demos" << endl;
	cout << "       blending result will be printed to stdout" << endl;
}

}

int main(int argc, char* argv[])
{
	using namespace std;
	if (argc < 3) {
		usage();
		return 0;
	}
	Graph g;
	g.loadRoadMap(move(ifstream(argv[1])));
	g.mergePath(move(ifstream(argv[2])));
	g.printGraph(cout);
}
