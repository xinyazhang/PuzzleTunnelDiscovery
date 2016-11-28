#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include "goctree.h"
#include <string>

using std::string;

template<int ND, typename FLOAT>
bool
coverage(const Eigen::Matrix<FLOAT, ND, 1>& state,  const Eigen::VectorXd& clearance, GOcTreeNode<ND, FLOAT> *node)
{
	typedef GOcTreeNode<ND, FLOAT> Node;
	// NOTE: check getClearanceCube for the exact meaning of clearance
	// vector.
	typename Node::Coord mins, maxs;
	node->getBV(mins, maxs);
	//std::cerr << "mins: " << mins.transpose() << " should > " << (state - clearance.segment<ND>(0)).transpose() << std::endl;
	//std::cerr << "maxs: " << maxs.transpose() << " should < " << (state + clearance.segment<ND>(0)).transpose() << std::endl;
	for (int i = 0; i < ND; i++) {
		if (mins(i) < state(i) - clearance(i))
			return false;
		if (maxs(i) > state(i) + clearance(i))
			return false;
	}
	return true;
}

template<int ND, typename FLOAT, typename CC>
GOcTreeNode<ND, FLOAT>*
buildOcTreeFromPath(const Geo& robot, const Geo& env, Path& path, CC &cc)
{
	typedef GOcTreeNode<ND, FLOAT> Node;
	typedef typename Node::Coord Coord;
	double t = 0.0;
	double end_t = path.T.size() - 1;
	Coord min, max;
	min << -100.0, -100.0, -100.0, 0.0, 0.0, 0.0;
	max <<  100.0,  100.0,  100.0, M_PI * 2, M_PI, M_PI * 2;
	Node* root = new Node(min, max, 0);
	for( ; t < end_t ; t += 1.0/12.0) {
		auto robot_transform_matrix = path.interpolate(robot, t);
		double mindist = cc.getDistance(robot_transform_matrix);
		auto clearance = cc.getClearanceCube(robot_transform_matrix, mindist);
		Coord state = Path::matrixToState(robot_transform_matrix);
		Node* current = root;
		//std::cerr << "Current clearance: " << clearance.transpose() << "\tMin distance: " << mindist << std::endl;
		while (current->getState() == Node::kCubeUncertain) {
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			//std::cerr << "Current: " << current << "\tNext: " << next << std::endl;
			if (coverage<ND, FLOAT>(state, clearance, next)) {
				next->setState(Node::kCubeFree);
			}
			current = next;
		}
		std::cerr << "t: " << t << std::endl;
	}

	return root;
}

template<int ND, typename FLOAT, typename CC>
void
buildOcTreeRecursive(GOcTreeNode<ND, FLOAT>* node, CC &cc)
{
	typedef GOcTreeNode<ND, FLOAT> Node;
	auto state = node->getMedian();
	auto robot_transform_matrix = Path::stateToMatrix(state);
	double mindist = cc.getDistance(robot_transform_matrix);
	if (mindist > FLOAT(0)) {
		auto clearance = cc.getClearanceCube(robot_transform_matrix, mindist);
		if (coverage<ND, FLOAT>(state, clearance, node))
			node->setState(Node::kCubeFree);
	} else {
		auto blockage = cc.getSolidCube(robot_transform_matrix, -mindist);
		if (coverage<ND, FLOAT>(state, blockage, node))
			node->setState(Node::kCubeFull);
	}
	if (node->getState() != Node::kCubeUncertain)
		return ;
	for (unsigned long index = 0; index < (1 << ND); index++) {
		typename Node::CubeIndex ci(index);
		auto subnode = node->getCube(ci);
		buildOcTreeRecursive(subnode, cc);
	}
}

template<int ND, typename FLOAT, typename CC>
GOcTreeNode<ND, FLOAT>*
buildOcTree(const Geo& robot, const Geo& env, CC &cc)
{
	typedef GOcTreeNode<ND, FLOAT> Node;
	typename Node::Coord min, max;
	min << -100.0, -100.0, -100.0, 0.0, 0.0, 0.0;
	max <<  100.0,  100.0,  100.0, M_PI * 2, M_PI, M_PI * 2;
	Node* root = new Node(min, max, 0);
	buildOcTreeRecursive(root, cc);

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
	cc.setC(-100,100);

	// We want template instantiation, but we don't want to run
	// the code.
	if (false) {
		auto root = buildOcTreeFromPath<6, double>(robot, env, path, cc);
		(void)root;
	} else {
		auto complete_root = buildOcTree<6, double>(robot, env, cc);
		(void)complete_root;
	}
	std::cout << "Done, press enter to exit" << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

	return 0;
}
