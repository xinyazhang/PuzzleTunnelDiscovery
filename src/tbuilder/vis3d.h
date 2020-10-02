/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef VIS2D_H
#define VIS2D_H

#include <goct/nullvisualizer.h>
#include "naiverenderer.h"
#include <vecio/arrayvecio.h>
#include <iostream>
#include <unordered_set>

// FIXME: find a way to avoid using macros
#define SHOW_ADJACENCY 0
#define SHOW_AGGADJACENCY 0
#define SHOW_AGGPATH 1

#define COUNT_NODES 1

#if COUNT_NODES
typedef NodeCounterVisualizer NaiveVisualizer3DBase;
#else
typedef NullVisualizer NaiveVisualizer3DBase;
#endif

/*
 * FIXME: lots of dup code with 2D Visualizer
 *        Lines related to ND were changed
 */
class NaiveVisualizer3D : public NaiveVisualizer3DBase {
private:
	static constexpr int ND = 3;
public:
	static void initialize()
	{
		NaiveVisualizer3DBase::initialize();
		aggpath_token = -1;
		rearmTimer();
	}

#if SHOW_ADJACENCY
	template<typename Node>
	static void visAdj(Node* node, Node* neighbor)
	{
		auto adj = build_line(node, neighbor);
		renderer_->addLine(adj);
	}
#endif
	template<typename Node>
	static void visPathSegment(Node* node, Node* neighbor)
	{
		auto adj = build_line(node, neighbor);
		renderer_->addLine(adj);
	}

#if SHOW_AGGPATH
	static void visAggPath(const std::vector<Eigen::VectorXd>& aggpath)
	{
		static int path_counter = 0;
		// std::cerr << "Aggressive path: " << aggpath << std::endl;
		if (aggpath_token > 0) {
			renderer_->removeDynamicLine(aggpath_token);
		}
		Eigen::MatrixXd adj;
		adj.resize(aggpath.size(), ND);
		for (size_t i = 0 ; i < aggpath.size(); i++) {
			adj.row(i) = aggpath[i];
		}
		aggpath_token = renderer_->addDynamicLine(adj);
		path_counter++;
		if (path_counter == 300) {
			for (const auto& v : aggpath) {
				std::cout << v.transpose() << ' ';
				std::cout << 0.0 << ' ' << 0.0 << ' ' << 0.0 << ' ' << 1.0;
				std::cout << std::endl;
			}
			std::cerr << "Some Agg Path was printed" << std::endl;
		}
		// pause();
	}
#endif

	static void pause()
	{
#if 1
		std::cerr << "Press enter to continue" << std::endl;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
#endif
	}
	
#if SHOW_AGGADJACENCY
	template<typename Node>
	static void visAggAdj(Node* node, Node* neighbor)
	{
		Eigen::MatrixXd adj = build_line(node, neighbor);
		int token = renderer_->addDynamicLine(adj);
		node->agg_line_tokens_.insert(token);
		neighbor->agg_line_tokens_.insert(token);
	}

	template<typename Node>
	static void withdrawAggAdj(Node* node)
	{
		for (auto token : node->agg_line_tokens_) {
			for (auto adj: node->getAggressiveAdjacency())
				adj->agg_line_tokens_.erase(token);
			renderer_->removeDynamicLine(token);
		}
	}

	struct Attribute {
		std::unordered_set<int> agg_line_tokens_;
	};
#endif

	template<typename Node>
	static void visSplit(Node* node)
	{
#if 0
		renderer_->addSplit(node->getMedian(), node->getMins(), node->getMaxs());
		// pause();
#endif
#if COUNT_NODES
		NaiveVisualizer3DBase::visSplit(node);
#endif
	}

	template<typename Node>
	static void visCertain(Node* node)
	{
#if 1 // Only visualize Full
		if (!node->atState(Node::kCubeFull))
			return ;
#endif
		renderer_->addCertain(node->getMedian(),
				node->getMins(),
				node->getMaxs(),
				node->getState() == Node::kCubeFree);
#if 0
		if (node->atState(Node::kCubeFull))
			pause();
#endif
	}

	static void setRenderer(NaiveRenderer* renderer) { renderer_ = renderer; }
private:
	static NaiveRenderer* renderer_;

	template<typename Node>
	static Eigen::MatrixXd build_line(Node* node, Node* neighbor)
	{
		Eigen::MatrixXd adj;
		adj.resize(2, ND);
		adj.row(0) = node->getMedian();
		adj.row(1) = neighbor->getMedian();
		return adj;
	}

#if SHOW_AGGPATH
	static int aggpath_token;
#endif
};

#endif
