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

template<int ND, typename FLOAT, typename Node>
bool
coverage(const Eigen::Matrix<FLOAT, ND, 1>& state,  const Eigen::VectorXd& clearance, Node *node)
{
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

#if 0
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
#endif

	void drawSplit(Node* node)
	{
		renderer_->addSplit(node->getMedian(), node->getMins(), node->getMaxs());
	}

	void drawCertain(Node* node)
	{
		renderer_->addCertain(node->getMedian(), node->getMins(), node->getMaxs(), node->getState() == Node::kCubeFree);
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
	init_p << 0.0, 0.0;
	goal_p << 10.0, 0.0;

	builder.setupInit(init_p);
	builder.setupGoal(goal_p);

	renderer->workerReady();

	builder.buildOcTree(cc);
	// std::cout << builder.buildPath();
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
