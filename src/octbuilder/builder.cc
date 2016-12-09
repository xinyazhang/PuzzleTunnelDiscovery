#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include "goctree.h"
#include "space.h"
#include <string>
#include <functional>
#include <deque>
#include <time.h>

using std::string;

template<int ND, typename FLOAT, typename Node>
bool
coverage(const Eigen::Matrix<FLOAT, ND, 1>& state,  const Eigen::VectorXd& clearance, Node *node)
{
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
	Node* root = Node::makeRoot(min, max);
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

template<int ND, typename FLOAT>
class OctreeBuilder {
public:
	typedef GOcTreeNode<ND, FLOAT> Node;
	typedef typename Node::Coord Coord;

	OctreeBuilder(const Coord& mins, const Coord& maxs, const Coord& res)
		:mins_(mins), maxs_(maxs), res_(res)
	{
	}

	template<typename CC>
	void buildOcTreeRecursive(Node* node, CC &cc, int depth)
	{
		auto state = node->getMedian();
		auto robot_transform_matrix = Path::stateToMatrix(state);
		double mindist = cc.getDistance(robot_transform_matrix);
		if (mindist > FLOAT(0)) {
			auto clearance = cc.getClearanceCube(robot_transform_matrix, mindist);
			//std::cerr << "Clearance for state: " << state.transpose() << " is " << clearance.transpose() << std::endl << "\tfrom " << mindist << std::endl;
			bool stop = coverage<ND, FLOAT>(state, res_, node) || coverage<ND, FLOAT>(state, clearance, node);
			if (stop)
				node->setState(Node::kCubeFree);
			else
				node->setState(Node::kCubeMixed);
		} else {
			auto blockage = cc.getSolidCube(robot_transform_matrix, -mindist);
			bool stop = coverage<ND, FLOAT>(state, res_, node) || coverage<ND, FLOAT>(state, blockage, node);
			if (stop)
				node->setState(Node::kCubeFull);
			else
				node->setState(Node::kCubeMixed);
		}
		if (node->isLeaf()) {
			fixed_volume_ += node->getVolume();
			if (::time(NULL) > last_time_ + 10) {
				double percent = (fixed_volume_ / total_volume_) * 100.0;
				std::cerr << "Progress: " << percent
				          << "\t(" << fixed_volume_
					  << " / " << total_volume_
					  << ")" << std::endl;
				last_time_ = ::time(NULL);
			}
			return ;
		}
		for (unsigned long index = 0; index < (1 << ND); index++) {
			typename Node::CubeIndex ci(index);
			auto subnode = node->getCube(ci);
			buildOcTreeRecursive(subnode, cc, depth + 1);
		}
	}

	template<typename CC>
	Node* buildOcTree(const Geo& robot, const Geo& env, CC &cc)
	{
		Coord min, max;
		Node* root = Node::makeRoot(mins_, maxs_);
		total_volume_ = root->getVolume();
		last_time_ = ::time(NULL);
		buildOcTreeRecursive(root, cc, 0);

		return root;
	}

private:
	Coord mins_, maxs_, res_;
	double fixed_volume_ = 0.0;
	double total_volume_;
	time_t last_time_;
};

template<int ND,
	 typename FLOAT,
	 typename CC,
	 typename Space = TranslationWithEulerAngleGroup<ND, FLOAT>
	>
class OctreePathBuilder {
	struct FindUnionAttribute {
		FindUnionAttribute* parent;
		FindUnionAttribute()
		{
			parent = this;
		}
		FindUnionAttribute* getSet() const
		{
			FindUnionAttribute* ret = this;
			if (parent != this)
				ret = parent->getSet();
			parent = ret;
			return ret;
		}
		void merge(FindUnionAttribute* other)
		{
			other->parent = this;
		}
	};
public:
#if 1
	typedef GOcTreeNode<ND, FLOAT, FindUnionAttribute> Node;
#else
	typedef GOcTreeNode<ND, FLOAT> Node;
#endif
	typedef typename Node::Coord Coord;

	OctreePathBuilder()
	{
	}

	void setupSpace(const Coord& mins, const Coord& maxs, const Coord& res)
	{
		mins_ = mins;
		maxs_ = maxs;
		res_ = res;
	}

	void setupInit(const Coord& initState)
	{
		istate_ = initState;
	}

	void setupGoal(const Coord& goalState)
	{
		gstate_ = goalState;
	}

	Node* determinizeCubeFromState(const Coord& state)
	{
		root_.reset(Node::makeRoot(mins_, maxs_));
		auto current = root_.get();
		auto robot_transform_matrix = Path::stateToMatrix(state);
		double mindist = cc_->getDistance(robot_transform_matrix);
		auto clearance = cc_->getClearanceCube(robot_transform_matrix, mindist);
		while (current->getState() == Node::kCubeUncertain) {
			std::cerr << "Current depth " << current->getDepth() << std::endl;
			auto children = split_cube(current);
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			if (coverage<ND, FLOAT>(state, clearance, next)) {
				next->setState(Node::kCubeFree);
			}
			current = next;
			// Add the remaining to the list.
			for (auto cube : children) {
				if (!cube->isLeaf() && cube != current)
					add_to_cube_list(cube);
			}
		}
		std::cerr << "Returning from " << __func__ << " current: " << current << std::endl;
		return current;
	}

	void add_neighbors(Node* node)
	{
		auto op = [=](int dim, int direct, std::vector<Node*>& neighbors)
		{
			for (auto neighbor: neighbors) {
				if (neighbor->getState() ==  Node::kCubeUncertain) {
					add_to_cube_list(neighbor);
#if VERBOSE
					std::cerr << __func__ << " : " << neighbor << std::endl;
#endif
				}
			}
		};
		contactors_op(node, op);
	}

	// Note: we don't set kCubeMixed because we want to add the mixed
	// cubes into queues.
	// 
	// In other words
	//      kCubeUncertain: cubes need to be splited
	//      kCubeUncertainPending: cubes in queue will be splited
	//      kCubeMixed: mixed cubes have been splited
	void check_clearance(Node* node)
	{
		auto state = node->getMedian();
		auto robot_transform_matrix = Path::stateToMatrix(state);
		double mindist = cc_->getDistance(robot_transform_matrix);
		if (mindist > FLOAT(0)) {
			auto clearance = cc_->getClearanceCube(robot_transform_matrix, mindist);
			bool stop = coverage<ND, FLOAT>(state, res_, node) ||
			            coverage<ND, FLOAT>(state, clearance, node);
			if (stop) {
				node->setState(Node::kCubeFree);
				fixed_volume_ += node->getVolume();
			}
		} else {
			auto blockage = cc_->getSolidCube(robot_transform_matrix, -mindist);
			bool stop = coverage<ND, FLOAT>(state, res_, node) ||
			            coverage<ND, FLOAT>(state, blockage, node);
			if (stop) {
				node->setState(Node::kCubeFull);
				fixed_volume_ += node->getVolume();
			}
		}
	}

	// Use Find-Union algorithm to merge adjacent free/blocked cubes.
	// Return true if free
	bool connect_neighbors(Node* node)
	{
		size_t ttlneigh = 0;
		auto op = [=,&ttlneigh](int dim, int direct, std::vector<Node*>& neighbors)
		{
			for (auto neighbor: neighbors) {
				if (neighbor->getState() == node->getState())
					node->merge(neighbor);
				ttlneigh++;
			}
		};
		contactors_op(node, op);
#if VERBOSE
		std::cerr << "Connected " << ttlneigh << " neighbors from Node (" << node->getMedian().transpose()
			  << ")\t depth: " << node->getDepth() << std::endl;
#endif
		
		return node->getState() == Node::kCubeFree;
	}

	void buildOcTree(const Geo& robot, const Geo& env, CC& cc)
	{
		robot_ = &robot; env_ = &env; cc_ = &cc;
		current_queue_ = 0;
		cubes_.clear();
		root_.reset();
		fixed_volume_ = 0.0;
		// Pre-calculate the initial clearance cube
		std::cerr << "Init: " << istate_.transpose() << std::endl;
		std::cerr << "Goal: " << gstate_.transpose() << std::endl;
		auto init_cube = determinizeCubeFromState(istate_);
		add_neighbors(init_cube);
		total_volume_ = root_->getVolume();
		while (true) {
			/*
			 * 1. Find a cube, called S, in the cubes_ list.
			 * 2. Split cube S.
			 * 2.1 This can be done through calling getCube
			 * 3.1 Check these newly created cubes, and connect determined
			 *      blocks with Union operator
			 * 3.2 Put undetermined blocks into cubes_ list
			 * 3.2.1 Prioritize larger (smaller depth_) cubes
			 * 3.2.2 This also includes neighbor's of S.
			 * 4. Check if determined cubes contains gstate_
			 * 4.1 If true, terminate
			 * 4.2 Otherwise, start from 1.
			 */
			auto to_split = pop_from_cube_list();
			auto children = split_cube(to_split);
			bool done = false;
			for (auto cube : children) {
				if (!cube->isLeaf()) {
					add_to_cube_list(cube);
					continue;
				}
				if (!connect_neighbors(cube))
					continue;
#if 0 // Do we really need connect kCubeFull?
				if (cube->getState() != Node::kCubeFree)
					continue;
#endif
				if (cube->getState() != init_cube->getState())
					continue;
				if (cube->isContaining(gstate_)) {
					done = true;
					break;
				}
			}
			if (done)
				break;
			if (timer_alarming()) {
				double percent = (fixed_volume_ / total_volume_) * 100.0;
				std::cerr << "Progress: " << percent
				          << "%\t(" << fixed_volume_
					  << " / " << total_volume_
					  << ")" << std::endl;
				rearm_timer();
			}
		}
	}

	Node* getRoot() { return root_.get(); }

private:
	std::vector<Node*> split_cube(Node* node)
	{
#if VERBOSE
		std::cerr << "Splitting (" << node->getMins().transpose()
		          << ")\t(" << node->getMaxs().transpose() << ")" << std::endl;
		std::cerr << "Splitting (" << node->getMedian().transpose()
		          << ")\t depth: " << node->getDepth() << std::endl;
#endif
		std::vector<Node*> ret;
		for (unsigned long index = 0; index < (1 << ND); index++) {
			typename Node::CubeIndex ci(index);
			ret.emplace_back(node->getCube(ci));
			check_clearance(ret.back());
		}
		node->setState(Node::kCubeMixed);
		//std::cerr << "\tResult: " << ret.size() << " child cubes" << std::endl;
		return ret;
	}

	// FIXME: oob check for empty queue
	Node* pop_from_cube_list()
	{
		int depth = current_queue_;
		while (cubes_[depth].empty())
			depth++;
		auto ret = cubes_[depth].front();
		cubes_[depth].pop_front();
		current_queue_ = depth;
		return ret;
	}

	void add_to_cube_list(Node* node)
	{
		if (node->getState() != Node::kCubeUncertain)
			return;
		int depth = node->getDepth();
		if (long(cubes_.size()) <= depth)
			cubes_.resize(depth+1);
		cubes_[depth].emplace_back(node);
		node->setState(Node::kCubeUncertainPending);
		current_queue_ = std::min(current_queue_, depth);
	}

	void contactors_op(Node* node,
			   std::function<void(int dim, int direct, std::vector<Node*>&)> op)
	{
		for (int dim = 0; dim < ND; dim++) {
			for (int direct = -1; direct <= 1; direct += 2) {
				auto neighbors = Node::getContactCubes(
						root_.get(),
						node,
						dim,
						direct,
						Space()
						);
				if (!neighbors.empty())
					op(dim, direct, neighbors);
			}
		}
	}

	bool timer_alarming() const
	{
		return (::time(NULL) > last_time_ + 10);
	}

	void rearm_timer()
	{
		last_time_ = ::time(NULL);
	}

	Coord mins_, maxs_, res_;
	Coord istate_, gstate_;
	const Geo *robot_, *env_;
	CC *cc_;
	std::unique_ptr<Node> root_;
	int current_queue_;
	std::vector<std::deque<Node*>> cubes_;
	time_t last_time_ = 0;
	double fixed_volume_;
	double total_volume_;
};

void press_enter()
{
	std::cout << "Done, press enter to exit" << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
}

int main(int argc, char* argv[])
{
#if 0
	string robotfn = "../res/alpha/alpha-1.2.org.obj";
	string envfn = "../res/alpha/alpha_env-1.2.org.obj";
	string pathfn = "../res/alpha/alpha-1.2.org.path";
#else
	string robotfn = "../res/simple/robot.obj";
	string envfn = "../res/simple/env.obj";
	string pathfn = "../res/simple/naive.path";
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
	cc.setC(-100,100);

	typename GOcTreeNode<6, double>::Coord min, max, res;
	min << -100.0, -100.0, -100.0, -M_PI/2.0,      0.0,      0.0;
	max <<  100.0,  100.0,  100.0,  M_PI/2.0, M_PI * 2, M_PI * 2;
	res = (max - min) / 20000.0;

	// We want template instantiation, but we don't want to run
	// the code.
	if (false) {
		auto root = buildOcTreeFromPath<6, double>(robot, env, path, cc);
		(void)root;
		press_enter();
	} else if (false) {
		//res << 0.001, 0.001, 0.001, M_PI/64.0, M_PI/32.0, M_PI/64.0;
		OctreeBuilder<6, double> builder(min, max, res);
		auto complete_root = builder.buildOcTree(robot, env, cc);
		(void)complete_root;
		press_enter();
	} else if (true) {
		OctreePathBuilder<6, double, decltype(cc)> builder;
		builder.setupSpace(min, max, res);
		double init_t = path.T.size() - 2;
		builder.setupInit(Path::matrixToState(path.interpolate(robot, init_t)));
		double end_t = path.T.size() - 1;
		builder.setupGoal(Path::matrixToState(path.interpolate(robot, end_t)));
		builder.buildOcTree(robot, env, cc);
		press_enter();
	}

	return 0;
}
