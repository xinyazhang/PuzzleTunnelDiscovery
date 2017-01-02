#ifndef GBUILDER_H
#define GBUILDER_H

#include "nullvisualizer.h"
#include "goctree.h"
#include <deque>
#include <algorithm>
#include <functional>
#include <queue>
#include <iostream>
#include <stdexcept>

/*
 * DFS: prioritize larger cubes connected to the initial cube
 * BFS: prioritize larger cubes globally.
 */
#ifndef ENABLE_DFS
#define ENABLE_DFS 1
#endif

template<int ND,
	 typename FLOAT,
	 typename CC,
	 typename Space,
	 typename Visualizer = NullVisualizer
	>
class GOctreePathBuilder {
	struct PathBuilderAttribute : public Visualizer::Attribute {
		//static constexpr auto kUnviewedDistance = ULONG_MAX;
		double distance; // = kUnviewedDistance;
		const PathBuilderAttribute* prev = nullptr;
		int epoch = -1;
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
	typedef GOcTreeNode<ND, FLOAT, FindUnionAttribute> Node;
	typedef typename Node::Coord Coord;
	typedef Visualizer VIS;

	static bool coverage(const Coord& state,
			    const Coord& clearance,
			    Node *node)
	{
		Coord mins, maxs;
		node->getBV(mins, maxs);
		for (int i = 0; i < ND; i++) {
			if (mins(i) < state(i) - clearance(i))
				return false;
			if (maxs(i) > state(i) + clearance(i))
				return false;
		}
		return true;
	}

	GOctreePathBuilder()
	{
	}

	void setupSpace(const Coord& mins, const Coord& maxs, const Coord& res)
	{
		mins_ = mins;
		maxs_ = maxs;
		res_ = res;
	}

	void setupInit(const Coord& initState) { istate_ = initState; }
	void setupGoal(const Coord& goalState) { gstate_ = goalState; }

	void buildOcTree(CC& cc)
	{
		init_builder(cc);
		VIS::initialize();
		VIS::rearmTimer();
		while (true) {
			auto to_split = pop_from_cube_list();
			auto children = split_cube(to_split);
			for (auto cube : children) {
				if (cube->getState() == Node::kCubeUncertain) {
					add_to_cube_list(cube);
				}
				connect_neighbors(cube);

				if (!cube->atState(Node::kCubeFree))
					continue;
#if ENABLE_DFS
				// From now we assume cube.state == free.
				if (cube->getSet() == init_cube_->getSet()) {
					auto firstorder = add_neighbors_to_list(cube);
					(void)firstorder;
					// Add second order neighbors.
					for (auto neighbor : firstorder)
						add_neighbors_to_list(neighbor);
					VIS::trackFurestCube(cube, init_cube_);
				}
#endif
			}
			if (goal_cube_ && goal_cube_->getSet() == init_cube_->getSet())
				break;
			if (VIS::timerAlarming()) {
				VIS::periodicalReport();
				std::cerr << "Fixed volume: " << fixed_volume_ << std::endl;
				VIS::rearmTimer();

				if (goal_cube_) {
					auto aggpath = buildNodePath(true);
					for (auto node : aggpath) {
						if (node->isDetermined())
							continue;
						add_to_oob_list(const_cast<Node*>(node));
					}
					VIS::visAggPath(convertNodePath(aggpath));
					if (aggpath.empty()) {
						std::cerr << "CANNOT FIND A PATH, EXITING\n";
						break;
					}
				}
			}
		}
	}

	std::vector<Eigen::VectorXd> buildPath(bool aggressive = false)
	{
		return convertNodePath(buildNodePath(aggressive));
	}

	std::vector<Eigen::VectorXd> convertNodePath(const std::vector<const Node*>& nodes)
	{
		std::vector<Eigen::VectorXd> ret;
		ret.reserve(nodes.size() + 2);
		ret.emplace_back(init_cube_->getMedian());
		for (const Node *node : nodes) {
			ret.emplace_back(node->getMedian());
		}
		ret.emplace_back(goal_cube_->getMedian());
		return ret;
	}

	std::vector<const Node*> buildNodePath(bool aggressive = false)
	{
		epoch_++;
		int epoch = epoch_;
		auto goal_cube = goal_cube_;

		init_cube_->distance = 0;
		init_cube_->prev = init_cube_; // Only circular one
		init_cube_->epoch = epoch;

		// priority_queue:
		//      cmp(top, other) always returns false.
		auto cmp = [](Node* lhs, Node* rhs) -> bool
			{ return lhs->distance < rhs->distance; };
		std::priority_queue<Node*, std::deque<Node*>, decltype(cmp)> Q(cmp);
		Q.push(init_cube_);

		bool goal_reached = false;
		auto loopf = [&Q, &goal_reached, epoch, goal_cube]
			(Node* adj, Node* tip) -> bool {
				if (adj->prev && adj->epoch == epoch) // No re-insert
					return false;
				adj->prev = tip;
				adj->distance = tip->distance +
					(tip->getMedian() - adj->getMedian()).norm();
				adj->epoch = epoch;
				Q.push(adj);
				if (adj == goal_cube) {
					goal_reached = true;
					return true;
				}
				return false;
			};
		while (!Q.empty() && !goal_reached) {
			auto tip = Q.top();
			Q.pop();
			for (auto adj : tip->getAdjacency())
				if (loopf(adj, tip))
					break;
			if (!goal_reached && aggressive)
				for (auto adj : tip->getAggressiveAdjacency())
					if (loopf(adj, tip))
						break;
		}
		if (!goal_reached)
			return {};
		std::vector<const Node*> ret;
		const Node* node = goal_cube_;
		while (node->prev != node) {
			ret.emplace_back(node);
			node = static_cast<const Node*>(node->prev);
		}
		std::reverse(ret.begin(), ret.end());
		return ret;
	}

protected:
	Node* determinize_cube(const Coord& state)
	{
		auto current = root_.get();

		while (!current->isDetermined()) {
			auto children = split_cube(current);
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			current = next;
			for (auto cube : children) {
#if !ENABLE_DFS
				// Add the remaining to the list.
				// Note: add_to_cube_list requires init_cube_ for DFS
				//       but init_cube_ is initialized by this
				//       function.
				if (cube->atState(Node::kCubeUncertain) && cube != current)
					add_to_cube_list(cube);
#endif
				connect_neighbors(cube);
			}
		}
		return current;
	}

	bool connect_neighbors(Node* node)
	{
		size_t ttlneigh = 0;
		auto op = [=,&ttlneigh](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
			for (auto neighbor: neighbors) {
				// TODO: WHY neighbors include itself?! CHECK
				// getContactCubes.
				if (node == neighbor)
					continue;
				if (neighbor->isDetermined() &&
				    neighbor->getState() == node->getState()) {
					node->merge(neighbor);
					Node::setAdjacency(node, neighbor);
					if (node->getState() == Node::kCubeFree) {
						VIS::visAdj(node, neighbor);
					}
				}
				if (Node::hasAggressiveAdjacency(node, neighbor)) {
					bool inserted;
					inserted = Node::setAggressiveAdjacency(node, neighbor);
					if (inserted) {
						VIS::visAggAdj(node, neighbor);
					}
				}
				ttlneigh++;
			}
			return false;
		};
		contactors_op(node, op);
		
		return node->getState() == Node::kCubeFree;
	}

	std::vector<Node*> add_neighbors_to_list(Node* node)
	{
		std::vector<Node*> ret;
		auto op = [=,&ret](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
			for (auto neighbor: neighbors) {
				if (neighbor->getState() ==  Node::kCubeUncertain) {
					if (add_to_cube_list(neighbor, false))
						ret.emplace_back(neighbor);
				}
			}
			return false;
		};
		contactors_op(node, op);
		return ret;
	}

	void check_clearance(Node* node)
	{
		auto state = node->getMedian();
		
		bool isfree;
		auto certain = cc_->getCertainCube(state, isfree);

		bool stop = coverage(state, res_, node) ||
			    coverage(state, certain, node);
		node->volume = node->getVolume();
		fixed_volume_ += node->getVolume();
		if (stop) {
			if (isfree) {
				node->setState(Node::kCubeFree);
				if (goal_cube_ == nullptr && node->isContaining(gstate_)) {
					goal_cube_ = node;
					std::cerr << "Goal cube cleared: " << *node << std::endl;
				}
			} else {
				node->setState(Node::kCubeFull);
			}
			VIS::visCertain(node);
		}
	}

	void init_builder(CC& cc)
	{
		cc_ = &cc;
		current_queue_ = 0;
		cubes_.clear();
		root_.reset(Node::makeRoot(mins_, maxs_));
		fixed_volume_ = 0.0;

		std::cerr << "Init: " << istate_.transpose() << std::endl;
		std::cerr << "Goal: " << gstate_.transpose() << std::endl;

		goal_cube_ = nullptr;
		init_cube_ = determinize_cube(istate_);
		std::cerr << "Init Cube: " << *init_cube_ << std::endl;
		add_neighbors_to_list(init_cube_);
	}

	std::vector<Node*> split_cube(Node* node)
	{
		node->setState(Node::kCubeMixed);
		VIS::visSplit(node);

		std::vector<Node*> ret;
		for (unsigned long index = 0; index < (1 << ND); index++) {
			typename Node::CubeIndex ci(index);
			ret.emplace_back(node->getCube(ci));
			check_clearance(ret.back());
		}
		VIS::withdrawAggAdj(node);
		node->cancelAggressiveAdjacency();
		
		return ret;
	}

	Node* pop_from_cube_list()
	{
		int depth = current_queue_;
		while (cubes_[depth].empty() && depth < int(cubes_.size()))
			depth++;
		if (depth >= int(cubes_.size()))
			throw std::runtime_error("CUBE LIST BECOMES EMPTY, TERMINATE");
		auto ret = cubes_[depth].front();
		cubes_[depth].pop_front();
		current_queue_ = depth;
		VIS::visPop(ret);
		return ret;
	}

	bool add_to_cube_list(Node* node, bool check_contacting_free = true)
	{
		if (node->getState() != Node::kCubeUncertain)
			return false;
		if (ENABLE_DFS &&
		    check_contacting_free &&
		    !is_contacting_free(node))
			return false;
		int depth = node->getDepth();
		if (long(cubes_.size()) <= depth)
			cubes_.resize(depth+1);
		cubes_[depth].emplace_back(node);
		node->setState(Node::kCubeUncertainPending);
		current_queue_ = std::min(current_queue_, depth);
		VIS::visPending(node);
		return true;
	}

	bool add_to_oob_list(Node* node)
	{
		cubes_[0].emplace_back(node);
		return true;
	}

	bool is_contacting_free(Node* node)
	{
		bool ret = false;
		auto op = [&ret, this](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
			for (auto neighbor: neighbors) {
				if (neighbor->isLeaf() &&
				    neighbor->isAggressiveFree() &&
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

	using ContactorFunctor = std::function<bool(int dim, int direct, std::vector<Node*>&)>;
	void contactors_op(Node* node,
			   ContactorFunctor op)
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
				if (neighbors.empty())
					continue;
				bool terminate = op(dim, direct, neighbors);
				if (terminate)
					return;
			}
		}
	}

	Coord mins_, maxs_, res_;
	Coord istate_, gstate_;
	CC *cc_;
	std::unique_ptr<Node> root_;
	int current_queue_;
	std::vector<std::deque<Node*>> cubes_;
	Node *init_cube_ = nullptr;
	Node *goal_cube_ = nullptr;
	int epoch_ = 0;
	double fixed_volume_;
};

#endif
