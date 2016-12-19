#include <omplaux/clearance.h>
#include <omplaux/path.h>
#include <omplaux/scene_bounding_box.h>
#include "goctree.h"
#include "space.h"
#include <string>
#include <functional>
#include <deque>
#include <queue>
#include <time.h>
#include <climits>

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
	struct PathBuilderAttribute {
		//static constexpr auto kUnviewedDistance = ULONG_MAX;
		double distance; // = kUnviewedDistance;
		const PathBuilderAttribute* prev = nullptr;
	};
	struct FindUnionAttribute : public PathBuilderAttribute {
		mutable const FindUnionAttribute* parent;
		mutable double volume = 0.0;

		FindUnionAttribute()
		{
			parent = this;
		}

		const FindUnionAttribute* getSet() const
		{
			const FindUnionAttribute* ret = this;
			if (parent != this)
				ret = parent->getSet();
			parent = ret;
			return ret;
		}

		void merge(FindUnionAttribute* other)
		{
			if (getSet() == other->getSet())
				return ;
			getSet()->volume += other->getSet()->volume;
			//other->getSet()->volume = 0.0;
			other->getSet()->parent = getSet();
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
				if (node->isContaining(gstate_)) {
					std::cerr << "Goal Cube Cleared (" << node->getMedian().transpose()
						<< ")\t depth: " << node->getDepth() << std::endl;
				}
				node->setState(Node::kCubeFree);

				node->volume = node->getVolume();
				fixed_volume_ += node->getVolume();
			}
		} else {
			auto blockage = cc_->getSolidCube(robot_transform_matrix, -mindist);
			bool stop = coverage<ND, FLOAT>(state, res_, node) ||
			            coverage<ND, FLOAT>(state, blockage, node);
			if (stop) {
				node->setState(Node::kCubeFull);

				node->volume = node->getVolume();
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
				if (neighbor->isLeaf()) {
					if (neighbor->getState() == node->getState()) {
#if 0
						std::cerr << "Merging: " << neighbor
							<< " to " << node
							<< std::endl;
						std::cerr << "\tBefore Volume: " << neighbor->getSet()->volume
							<< " and " << node->getSet()->volume
							<< std::endl;
#endif
						node->merge(neighbor);
						Node::setAdjacency(node, neighbor);
#if 0
						std::cerr << "\tAfter Volume: " << neighbor->getSet()->volume
							<< " and " << node->getSet()->volume
							<< std::endl;
#endif
					}
				}
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
#if 0
		auto goal_cube = determinizeCubeFromState(gstate_);
		add_neighbors(goal_cube);
#else
		decltype(init_cube) goal_cube = nullptr;
#endif
		total_volume_ = root_->getVolume();
		double max_cleared_distance = 0.0;
		Eigen::VectorXd max_cleared_median;
		// TODO: use DFS instead of BFS
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
				if (cube->getState() != Node::kCubeFree)
					continue;
				if (!goal_cube && cube->isContaining(gstate_)) {
					goal_cube = cube;
				}

				// Track the furthest reachable cube.
				if (cube->getSet() == init_cube->getSet()) {
					Eigen::VectorXd dis = cube->getMedian() - init_cube->getMedian();
					double disn = dis.block<3,1>(0,0).norm();
					if (disn > max_cleared_distance) {
						max_cleared_distance = disn;
						max_cleared_median = dis;
					}
				}
			}
			if (goal_cube && goal_cube->getSet() == init_cube->getSet())
				break;
			if (timer_alarming()) {
				double percent = (fixed_volume_ / total_volume_) * 100.0;
				std::cerr << "Progress: " << percent
				          << "%\t(" << fixed_volume_
					  << " / " << total_volume_
					  << ")\tMax cleared distance: " << max_cleared_distance
					  << "\tcube median: " << max_cleared_median.transpose()
					  << std::endl;
				std::cerr << "\tInit Set: " << init_cube->getSet()
					  << "\tInit Volume: " << init_cube->getSet()->volume
					  << std::endl;
				if (goal_cube)
					std::cerr << "\tGoal Set: " << goal_cube->getSet()
						  << "\tGoal Volume: " << goal_cube->getSet()->volume
						  << std::endl;
				rearm_timer();
			}
		}
		init_cube_ = init_cube;
		goal_cube_ = goal_cube;
	}

	Node* getRoot() { return root_.get(); }

	std::vector<Eigen::VectorXd> buildPath()
	{
		init_cube_->distance = 0;
		init_cube_->prev = init_cube_; // Only circular one
		auto cmp = [](Node* lhs, Node* rhs) -> bool { return lhs->distance < rhs->distance; };
		// priority_queue:
		//      cmp(top, other) always returns false.
		std::priority_queue<Node*, std::deque<Node*>, decltype(cmp)> Q(cmp);
		Q.push(init_cube_);
		bool goal_reached = false;
		while (!Q.empty() && !goal_reached) {
			auto tip = Q.top();
			Q.pop();
			for (auto adj : tip->getAdjacency()) {
				if (adj->prev) // No re-insert
					continue;
				adj->prev = tip;
				adj->distance = tip->distance + 1.0/pow(2.0, tip->getDepth());
				Q.push(adj);
				if (adj == goal_cube_) {
					goal_reached = true;
					break;
				}
			}
		}
		std::vector<Eigen::VectorXd> ret;
		const Node* node = goal_cube_;
		ret.emplace_back(Path::stateToPath<double>(gstate_));
		while (node->prev != node) {
			ret.emplace_back(Path::stateToPath<double>(node->getMedian()));
			node = static_cast<const Node*>(node->prev);
		}
		ret.emplace_back(Path::stateToPath<double>(node->getMedian()));
		ret.emplace_back(Path::stateToPath<double>(istate_));
		std::reverse(ret.begin(), ret.end());
		return ret;
	}

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
		return (::time(NULL) > last_time_); // Time interval
	}

	void rearm_timer()
	{
		last_time_ = ::time(NULL);
	}

	Coord mins_, maxs_, res_; Coord istate_, gstate_;
	const Geo *robot_, *env_;
	CC *cc_;
	std::unique_ptr<Node> root_;
	int current_queue_;
	std::vector<std::deque<Node*>> cubes_;
	time_t last_time_ = 0;
	double fixed_volume_;
	double total_volume_;
	Node *init_cube_ = nullptr;
	Node *goal_cube_ = nullptr;
};

void press_enter()
{
	std::cerr << "Done, press enter to exit" << std::endl;
	std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
}

// FIXME: put this into some lib
std::ostream& operator<<(std::ostream& fout, const std::vector<Eigen::VectorXd>& milestones)
{
	for(const auto& m : milestones) {
		fout << m.transpose() << std::endl;
	}
	return fout;
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
	min << bbmin, bbmin, bbmin, M_PI*0.0, -M_PI/2.0, M_PI*0.0;
	max << bbmax, bbmax, bbmax, M_PI*2.0,  M_PI/2.0, M_PI*2.0;
	std::cerr << "Bounding box\n"
	          << "\tmin: " << min.transpose() << "\n"
		  << "\tmax: " << max.transpose() << std::endl;
	cc.setC(bbmin, bbmax);
#endif
	res = (max - min) / 20000.0; // FIXME: how to calculate a resolution?

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
		//double init_t = 0.0;
		double init_t = path.T.size() - 2;
		builder.setupInit(Path::matrixToState(path.interpolate(robot, init_t)));
		double end_t = path.T.size() - 1;
		builder.setupGoal(Path::matrixToState(path.interpolate(robot, end_t)));
		builder.buildOcTree(robot, env, cc);
		std::cout << builder.buildPath();
		press_enter();
	}

	return 0;
}
