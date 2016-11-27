#include "clearance.h"
#include "goctree.h"

template<int ND, typename FLOAT, typename CC>
GOcTreeNode<ND, FLOAT>*
buildOcTree(const Geo& robot, const Geo& env, Path& path, CC &cc)
{
	typedef GOcTreeNode<ND, FLOAT> Node;
	double t = 0.0;
	double end_t = path.T.size();
	Node::Coord min, max;
	min << -100.0, -100.0, -100.0, 0.0, 0.0, 0.0;
	max <<  100.0,  100.0,  100.0, M_PI * 2, M_PI, M_PI * 2;
	Node* root = new Node(min, max, 0);
	for( ; t < end_t ; t += 1.0/12.0) {
		robot_transform_matrix = path.interpolate(robot, t);
		double mindist = cc.getDistance(robot_transform_matrix);
		auto clearance = cc.getClearanceCube(robot_transform_matrix, mindist);
		Coord state = Path::matrixToState(robot_transform_matrix);
		Node* current = root;
		while (current->getState() == kCubeUncertain) {
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			if (coverage(state, next, clearance)) {
				next->setState(Node::kCubeFree);
			}
			current = next;
		}
	}

	return root;
}

int main(int argc, char* argv[])
{
	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
	Geo robot, env;
	Path path;

	robot.read(robotfn);
	env.read(envfn);
	path.readPath(pathfn);
	robot.center << 16.973146438598633, 1.2278236150741577, 10.204807281494141; // From OMPL.app, no idea how they get this.

	ClearanceCalculator<fcl::OBBRSS<double>> cc(robot, env);
	auto root = buildOcTree<6, double>(robot, env, path, cc);
}
