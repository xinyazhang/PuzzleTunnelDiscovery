/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef GBUILDER_H
#define GBUILDER_H

#include "nullvisualizer.h"
#include "goctree.h"
#include <deque>
#include <thread>
#include <algorithm>
#include <functional>
#include <queue>
#include <iostream>
#include <stdexcept>
#include <boost/heap/priority_queue.hpp>

/*
 * DFS: prioritize larger cubes connected to the initial cube
 * BFS: prioritize larger cubes globally.
 */
#ifndef ENABLE_DFS
#define ENABLE_DFS 1
#endif

#ifndef PRIORITIZE_SHORTEST_PATH
#define PRIORITIZE_SHORTEST_PATH 0
#endif

#ifndef ENABLE_DIJKSTRA
#define ENABLE_DIJKSTRA 0
#endif

#if !PRIORITIZE_SHORTEST_PATH
#define PRIORITIZE_CLEARER_CUBE 1
#endif

template<int ND,
	 typename FLOAT,
	 typename CC,
	 typename Space,
	 typename Visualizer = NullVisualizer
	>
class GOctreePathBuilder {
	struct PathBuilderAttribute : public Visualizer::Attribute {
#ifdef ENABLE_DIJKSTRA
		double distance;
#else
		int distance;
#endif
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
#if !PRIORITIZE_CLEARER_CUBE
	using NodeAttribute = FindUnionAttribute;
#else
	struct NodeDistanceAttribute : public FindUnionAttribute {
		double certain_ratio;
	};
	using NodeAttribute = NodeDistanceAttribute;
#endif
public:
	/*
	 * Helper functions
	 */
	typedef GOcTreeNode<ND, FLOAT, NodeAttribute> Node;
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

	/*
	 * Tree building
	 */
	void buildOcTree(CC& cc)
	{
		init_builder(cc);
		VIS::initialize();
		VIS::rearmTimer();
		bool check_path = false;
		std::unordered_set<Node*> path_nodes;
		while (true) {
#if PRIORITIZE_SHORTEST_PATH
			check_path = isCubeListEmpty();
#endif
			if (check_path && goal_cube_) {
				path_nodes.clear();
				auto aggpath = buildNodePath(true);
				for (auto node : aggpath) {
					if (node->isDetermined())
						continue;
					add_to_cube_list(const_cast<Node*>(node));
					path_nodes.emplace(node);
				}
				VIS::visAggPath(convertNodePath(aggpath));
				if (aggpath.empty()) {
					std::cerr << "CANNOT FIND A PATH, EXITING\n";
					break;
				}
			}

			auto to_split = pop_from_cube_list();
			auto children = split_cube(to_split);
			bool direct_node = (path_nodes.find(to_split) != path_nodes.end());
			for (auto cube : children) {
#if !PRIORITIZE_SHORTEST_PATH
				if (cube->getState() == Node::kCubeUncertain) {
					add_to_cube_list(cube);
				}
#endif
				connect_neighbors(cube);

				if (direct_node && cube->atState(Node::kCubeFull))
					add_neighbors_to_list(cube, true);

				if (!cube->atState(Node::kCubeFree)) {
					// Full cube may have full neighbors.
					continue;
				}
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
				std::cerr << "Fixed volume: " << fixed_volume_ << "\tDeepest level: " << getDeepestLevel() << std::endl;
				VIS::rearmTimer();

				check_path = true;
			}
		}
		stop_builder();
	}

	std::vector<Eigen::VectorXd> buildPath(bool aggressive = false)
	{
		return convertNodePath(buildNodePath(aggressive));
	}

	std::vector<Eigen::VectorXd> convertNodePath(const std::vector<Node*>& nodes)
	{
		std::vector<Eigen::VectorXd> ret;
		ret.reserve(nodes.size());
		// ret.emplace_back(istate_);
		for (const Node *node : nodes) {
			ret.emplace_back(node->getMedian());
		}
		// ret.emplace_back(gstate_);
		return ret;
	}

#if ENABLE_DIJKSTRA
	/*
	 * Dijkstra with immutable priority queue.
	 * 
	 * epoch is used to denote the status of the node:
	 *      fresh epoch: 
	 *              Node::distance and Node::prev is valid
	 *      fin epoch:
	 *              Node::distance and Node::prev is final
	 *      others:
	 *              Both of them are uninitialized or the legacy from
	 *              previous epoch.
	 *
	 * Each node may have multiple copies in Q, since std::priority_queue
	 * is immutable.
	 */
	std::vector<Node*> buildNodePath(bool aggressive = false)
	{
		epoch_++;
		int freshepoch = epoch_;
		epoch_++;
		int finepoch = epoch_;
		auto goal_cube = goal_cube_;

		init_cube_->distance = 0;
		init_cube_->prev = init_cube_; // Only circular one

		// priority_queue:
		//      cmp(top, other) always returns false.
		auto cmp = [](Node* lhs, Node* rhs) -> bool
			{ return lhs->distance > rhs->distance; };
		std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> Q(cmp);
		Q.push(init_cube_);

		bool goal_reached = false;
		auto loopf = [&Q, &goal_reached, freshepoch, finepoch, goal_cube]
			(Node* adj, Node* tip) -> bool {
				bool fresh = (adj->epoch != freshepoch && adj->epoch != finepoch);
				bool do_update = false;
				// double w = (tip->getMedian() - adj->getMedian()).norm();
#if PRIORITIZE_CLEARER_CUBE
				double w = 1.0/adj->certain_ratio; // Perfer cubes further from obs
#else
				double w = 1.0; // prefer larger cube.
#endif
				double dist = tip->distance + w;
				do_update = fresh ? true : (adj->distance > dist);

				if (do_update) {
					adj->prev = tip;
					adj->distance = dist;
				}
				if (fresh)
					adj->epoch = freshepoch;
				Q.push(adj);
				return false;
			};

		while (!Q.empty() && !goal_reached) {
			auto tip = Q.top();
			Q.pop();
			if (tip->epoch == finepoch) // We may have duplicated nodes in Q.
				continue;
			tip->epoch = finepoch;
			if (tip == goal_cube) {
				goal_reached = true;
				break;
			}
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
		std::vector<Node*> ret;
		Node* node = goal_cube_;
		while (node->prev != node) {
			ret.emplace_back(node);
			node = const_cast<Node*>(static_cast<const Node*>(node->prev));
		}
		std::reverse(ret.begin(), ret.end());
		return ret;
	}
#else
	/*
	 * Simple BFS path builder
	 */
	std::vector<Node*> buildNodePath(bool aggressive = false)
	{
		epoch_++;
		int finepoch = epoch_;
		auto goal_cube = goal_cube_;
		bool goal_reached = false;

		init_cube_->distance = 0;
		init_cube_->prev = init_cube_; // Only circular one
		init_cube_->epoch = finepoch;

		std::queue<Node*> Q;
		Q.push(init_cube_);

		auto loopf = [&Q, &goal_reached, finepoch, goal_cube]
			(Node* adj, Node* tip) -> bool {
				if (adj->epoch == finepoch)
					return false;
				adj->epoch = finepoch;
				adj->distance = tip->distance + 1;
				adj->prev = tip;
				Q.push(adj);
				if (goal_cube == adj) {
					goal_reached = true;
					return true;
				}
				return false;
			};

		while (!Q.empty() && !goal_reached) {
			auto tip = Q.front();
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
		std::vector<Node*> ret;
		Node* node = goal_cube_;
		while (node->prev != node) {
			ret.emplace_back(node);
			node = const_cast<Node*>(static_cast<const Node*>(node->prev));
		}
		std::reverse(ret.begin(), ret.end());
		return ret;
	}
#endif

	unsigned getDeepestLevel()
	{
#if !PRIORITIZE_SHORTEST_PATH
		return cubes_.size();
#else
		return max_depth_;
#endif
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
#if PRIORITIZE_SHORTEST_PATH
		goal_cube_ = determinize_cube(gstate_);
		cubes_.resize(1);
		std::cerr << "Goal Cube: " << *goal_cube_ << std::endl;
#endif
		add_neighbors_to_list(init_cube_);

		launch_wall_finder();
	}

	Node* determinize_cube(const Coord& state)
	{
		auto current = root_.get();

		while (!current->isDetermined()) {
			auto children = split_cube(current);
			auto ci = current->locateCube(state);
			Node* next = current->getCube(ci);
			current = next;
			for (auto cube : children) {
#if !ENABLE_DFS && !PRIORITIZE_SHORTEST_PATH
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

protected:

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

	std::vector<Node*> add_neighbors_to_list(Node* node,
			bool enforce = false)
	{
		// Default policy of PRIORITIZE_SHORTEST_PATH
		//   Only add cubes on the shortest path
		// 
		// Set enforce to true to override this behavior.
#if PRIORITIZE_SHORTEST_PATH
		if (!enforce)
			return {};
#endif
		std::vector<Node*> ret;
		auto op = [=,&ret](int dim, int direct, std::vector<Node*>& neighbors) -> bool
		{
			for (auto neighbor: neighbors) {
				if (neighbor->getState() !=  Node::kCubeUncertain)
					continue;
				if (enforce && neighbor->getDepth() > node->getDepth())
					continue;
				if (add_to_cube_list(neighbor, false))
					ret.emplace_back(neighbor);
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

#if 0
		bool stop = coverage(state, res_, node) ||
			    coverage(state, certain, node);
#else
		bool stop = coverage(state, certain, node);
#endif
		node->volume = node->getVolume();
#if PRIORITIZE_CLEARER_CUBE
		node->certain_ratio = certain(0) / res_(0);
#endif
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

	void stop_builder()
	{
		stop_wall_finder();
	}

	/*
	 * Note: split_cube is designed can be called multiple times on the
	 *       same cube
	 */
	std::vector<Node*> split_cube(Node* node)
	{
		bool repeated = node->atState(Node::kCubeMixed);
		node->setState(Node::kCubeMixed);

		if (!repeated)
			VIS::visSplit(node);

		std::vector<Node*> ret;
		for (unsigned long index = 0; index < (1 << ND); index++) {
			typename Node::CubeIndex ci(index);
			ret.emplace_back(node->getCube(ci));
			check_clearance(ret.back());
		}
		VIS::withdrawAggAdj(node);
		node->cancelAggressiveAdjacency();
#if PRIORITIZE_SHORTEST_PATH
		max_depth_ = std::max(node->getDepth() + 1, max_depth_);
#endif
		return ret;
	}

	Node* pop_from_cube_list()
	{
		int depth = current_queue_;
		while (cubes_[depth].empty() && depth < int(cubes_.size()))
			depth++;
		if (depth >= int(cubes_.size()))
			throw std::runtime_error("CUBE LIST BECOMES EMPTY, TERMINATE");
#if !PRIORITIZE_CLEARER_CUBE
		auto ret = cubes_[depth].front();
		cubes_[depth].pop_front();
#else
		auto ret = cubes_[depth].top();
		cubes_[depth].pop();
#endif
		current_queue_ = depth;
		VIS::visPop(ret);
		return ret;
	}

	bool isCubeListEmpty()
	{
		for (const auto& list: cubes_)
			if (!list.empty())
				return false;
		return true;
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
#if !PRIORITIZE_CLEARER_CUBE
		cubes_[depth].emplace_back(node);
#else
		cubes_[depth].push(node);
#endif
		node->setState(Node::kCubeUncertainPending);
		current_queue_ = std::min(current_queue_, depth);
		VIS::visPending(node);
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

#if !PRIORITIZE_CLEARER_CUBE
	using PerDepthQ = std::deque<Node*>;
#else
	struct ClearerNode {
		bool operator() (Node* lhs, Node* rhs) {
			return lhs->certain_ratio > rhs->certain_ratio;
		}
	};
	using PerDepthQ = std::priority_queue<Node*, std::vector<Node*>, ClearerNode>;
#endif

	Coord mins_, maxs_, res_;
	Coord istate_, gstate_;
	CC *cc_;
	std::unique_ptr<Node> root_;
	int current_queue_;
	std::vector<PerDepthQ> cubes_;
	Node *init_cube_ = nullptr;
	Node *goal_cube_ = nullptr;
	int epoch_ = 0;
	double fixed_volume_;
#if PRIORITIZE_SHORTEST_PATH
	unsigned max_depth_ = 0;
#endif
	std::unique_ptr<std::thread> wall_finder_thread_;
	bool wall_finder_exiting_;

	void launch_wall_finder()
	{
		wall_finder_exiting_ = false;
		if (!wall_finder_thread_)
			wall_finder_thread_.reset(new std::thread([this](){this->wall_finder();}));
	}

	void stop_wall_finder()
	{
		wall_finder_exiting_ = true;
		if (wall_finder_thread_)
			wall_finder_thread_->join();
		wall_finder_thread_.reset();
	}

	// Another thread to detect walls?
	void wall_finder()
	{
	}
};

#endif
