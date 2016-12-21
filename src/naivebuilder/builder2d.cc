#include "naiveclearance.h"
#include "naivespace.h"
#include "naiverenderer.h"
#include "goctree.h"
#include <string>
#include <functional>
#include <deque>
#include <queue>
#include <time.h>
#include <climits>

using std::string;

void press_enter();

template<int ND, typename FLOAT, typename Node>
bool
coverage(const Eigen::Matrix<FLOAT, ND, 1>& state,  const Eigen::VectorXd& clearance, Node *node)
{
	typename Node::Coord mins, maxs;
	node->getBV(mins, maxs);
#if 0
	std::cerr << "mins: " << mins.transpose() << " should > " << (state - clearance.segment<ND>(0)).transpose() << std::endl;
	std::cerr << "maxs: " << maxs.transpose() << " should < " << (state + clearance.segment<ND>(0)).transpose() << std::endl;
#endif
	for (int i = 0; i < ND; i++) {
		if (mins(i) < state(i) - clearance(i))
			return false;
		if (maxs(i) > state(i) + clearance(i))
			return false;
	}
	return true;
}

template<int ND,
	 typename FLOAT,
	 typename CC,
	 typename Space = TranslationOnlySpace<ND, FLOAT>
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

	OctreePathBuilder(NaiveRenderer* renderer)
		:renderer_(renderer)
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

		bool isfree;
		auto certain = cc_->getCertainCube(state, isfree);

		while (current->getState() == Node::kCubeUncertain) {
			std::cerr << "Current depth " << current->getDepth() << std::endl;
			auto children = split_cube(current);
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			if (coverage<ND, FLOAT>(state, certain, next)) {
				if (isfree)
					next->setState(Node::kCubeFree);
				else
					next->setState(Node::kCubeFull);
				drawCertain(next);
			}
			current = next;
#if 0 // Not necessary in DFS
			// Add the remaining to the list.
			for (auto cube : children) {
				if (!cube->isLeaf() && cube != current)
					add_to_cube_list(cube);
			}
#endif
		}
		std::cerr << "Returning from " << __func__ << " current: " << current << std::endl;
		return current;
	}

	void add_neighbors(Node* node)
	{
#if VERBOSE
		std::cerr << "BEGIN " << __func__ << std::endl;
#endif
		auto op = [=](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
			for (auto neighbor: neighbors) {
#if VERBOSE
				std::cerr << "checking neighbor: " << *neighbor << std::endl;
#endif
				if (neighbor->getState() ==  Node::kCubeUncertain) {
					add_to_cube_list(neighbor, false);
#if VERBOSE
					std::cerr << __func__ << " : " << neighbor << std::endl;
#endif
				}
			}
			return false;
		};
		contactors_op(node, op);
#if VERBOSE
		std::cerr << "TERMINATE " << __func__ << std::endl;
#endif
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
		
		bool isfree;
		auto certain = cc_->getCertainCube(state, isfree);

		bool stop = coverage<ND, FLOAT>(state, res_, node) ||
			    coverage<ND, FLOAT>(state, certain, node);
		node->volume = node->getVolume();
		fixed_volume_ += node->getVolume();
		if (stop) {
			if (node->isContaining(gstate_)) {
				std::cerr << "Goal Cube Cleared (" << node->getMedian().transpose()
					<< ")\t depth: " << node->getDepth() << std::endl;
			}
			if (isfree) {
				node->setState(Node::kCubeFree);
			} else {
				node->setState(Node::kCubeFull);
			}
			drawCertain(node);
		}
	}

	// Use Find-Union algorithm to merge adjacent free/blocked cubes.
	// Return true if free
	bool connect_neighbors(Node* node)
	{
		size_t ttlneigh = 0;
		auto op = [=,&ttlneigh](int dim, int direct, std::vector<Node*>& neighbors) -> bool
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
			return false;
		};
		contactors_op(node, op);
#if VERBOSE
		std::cerr << "Connected " << ttlneigh << " neighbors from Node (" << node->getMedian().transpose()
			  << ")\t depth: " << node->getDepth() << std::endl;
#endif
		
		return node->getState() == Node::kCubeFree;
	}

	void buildOcTree(CC& cc)
	{
		cc_ = &cc;
		current_queue_ = 0;
		cubes_.clear();
		root_.reset();
		fixed_volume_ = 0.0;
		// Pre-calculate the initial clearance cube
		std::cerr << "Init: " << istate_.transpose() << std::endl;
		std::cerr << "Goal: " << gstate_.transpose() << std::endl;
		auto init_cube = determinizeCubeFromState(istate_);
		init_cube_ = init_cube;
		add_neighbors(init_cube);
		std::cerr << "add_neighbors DONE\n";
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
				if (cube->getState() == Node::kCubeUncertain) {
#if VERBOSE
					std::cerr << "\t= Try to add child to list "
						  << *cube
						  << std::endl;
#endif
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
				// From now we assume cube.state == free.
				if (!goal_cube && cube->isContaining(gstate_)) {
					goal_cube = cube;
					goal_cube_ = goal_cube;
				}
				if (cube->getSet() == init_cube->getSet()) {
					add_neighbors(cube);

					// Track the furthest reachable cube.
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
				verbose(init_cube, goal_cube, max_cleared_distance, max_cleared_median);
				rearm_timer();
			}
		}
	}

	void verbose(Node* init_cube, Node* goal_cube, double max_cleared_distance, const Eigen::VectorXd& max_cleared_median)
	{
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
	}

	Node* getRoot() { return root_.get(); }

#if 1
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
		ret.emplace_back(gstate_);
		while (node->prev != node) {
			ret.emplace_back(node->getMedian());
			node = static_cast<const Node*>(node->prev);
		}
		ret.emplace_back(node->getMedian());
		ret.emplace_back(istate_);
		std::reverse(ret.begin(), ret.end());
		return ret;
	}
#endif

	void drawSplit(Node* node)
	{
		//std::cerr << "Adding split: " << node->getMins() << " - " << node->getMaxs() << std::endl;
		renderer_->addSplit(node->getMedian(), node->getMins(), node->getMaxs());
		press_enter();
	}

	void drawCertain(Node* node)
	{
		std::cerr << "Adding certain: " << node->getMins().transpose() << " - " << node->getMaxs().transpose() << "\tCenter: " << node->getMedian().transpose() << std::endl;
		renderer_->addCertain(node->getMedian(), node->getMins(), node->getMaxs(), node->getState() == Node::kCubeFree);
		press_enter();
	}

	void setupRenderer(NaiveRenderer* renderer)
	{
		renderer_ = renderer;
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
		drawSplit(node);

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

	bool contacting_free(Node* node)
	{
		bool ret = false;
#if VERBOSE
		std::cerr << "Checking contacting_free: " << *node  << std::endl;
#endif
		auto op = [&ret, this](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
#if VERBOSE
			std::cerr << "dim: " << dim << "\tdirect: " << direct << "\t# neighbors: " << neighbors.size() << std::endl;
#endif
			for (auto neighbor: neighbors) {
#if VERBOSE
				std::cerr << "\tNeighbor: " << *neighbor << std::endl;
				std::cerr << "\tLeaf? " << neighbor->isLeaf() << "\tState? " << neighbor->getState() << std::endl;
#endif
				if (neighbor->isLeaf() &&
				    neighbor->getState() == Node::kCubeFree &&
				    neighbor->getSet() == init_cube_->getSet()) {
					ret = true;
					return true;
				}
			}
			return false;
		};
		contactors_op(node, op);

		return ret;
	}

	void add_to_cube_list(Node* node, bool do_check = true)
	{
		if (node->getState() != Node::kCubeUncertain)
			return;
		if (do_check && !contacting_free(node))
			return;
		int depth = node->getDepth();
		if (long(cubes_.size()) <= depth)
			cubes_.resize(depth+1);
		cubes_[depth].emplace_back(node);
#if VERBOSE
		std::cerr << "-----Add one into list\n";
#endif
		node->setState(Node::kCubeUncertainPending);
		current_queue_ = std::min(current_queue_, depth);
	}

	void contactors_op(Node* node,
			   std::function<bool(int dim, int direct, std::vector<Node*>&)> op)
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
				if (!neighbors.empty()) {
					bool terminate = op(dim, direct, neighbors);
					if (terminate)
						return;
				}
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

	NaiveRenderer* renderer_;
	Coord mins_, maxs_, res_; Coord istate_, gstate_;
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
	std::cerr << "Press enter to continue" << std::endl;
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

int worker(NaiveRenderer* renderer)
{
	string envfn = "../res/simple/Torus.obj";
	Geo env;
	env.read(envfn);
	env.V.block(0, 2, env.V.rows(), 1) *= 0.0001;

	renderer->setEnv(&env);
	NaiveClearance cc(env);

	using Builder = OctreePathBuilder<2, double, decltype(cc)>;
	using Coord = typename Builder::Coord;
	Coord min, max, res;

	double tmin = -10;
	double tmax = 10;
	min << tmin, tmin;
	max << tmax, tmax;
	cc.setC(tmin, tmax);

	res = (max - min) / 20000.0; // FIXME: how to calculate a resolution?

	// We want template instantiation, but we don't want to run
	// the code.
	Builder builder(renderer);
	builder.setupSpace(min, max, res);
	builder.setupRenderer(renderer);
	Coord init_p, goal_p;
	init_p << -1.0, -1.0;
	goal_p << -9.0, 0.0;

	builder.setupInit(init_p);
	builder.setupGoal(goal_p);

	renderer->workerReady();

	builder.buildOcTree(cc);
	auto path = builder.buildPath();
	std::cerr << path << std::endl;
	Eigen::MatrixXd np;
	np.resize(path.size(), path.front().size() + 1);
	for (size_t i = 0; i < path.size(); i++) {
		np.row(i) = path[i];
		np(i, 2) = 2.0;
	}
	renderer->addLine(np);
	std::cerr << "Done\n";
	press_enter();

	return 0;
}

int main(int argc, char* argv[])
{
	Naive2DRenderer render;
	render.init();
	render.launch_worker(worker);
	return render.run();
}
