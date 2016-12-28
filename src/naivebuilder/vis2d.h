#ifndef VIS2D_H
#define VIS2D_H

#include "nullvisualizer.h"
#include "naiverenderer.h"
#include <vecio/arrayvecio.h>
#include <iostream>
#include <set>

// FIXME: find a way to avoid using macros
#define SHOW_ADJACENCY 1
#define SHOW_AGGADJACENCY 1
#define SHOW_AGGPATH 1

/* 2D Visualizer */
class NaiveVisualizer : public NullVisualizer {
private:
	static constexpr int ND = 2;
public:
	static void initialize()
	{
#if SHOW_AGGPATH
		aggpath_token = -1;
#endif
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

#if SHOW_AGGPATH
	static void visAggPath(const std::vector<Eigen::VectorXd>& aggpath)
	{
		std::cerr << "Aggressive path: " << aggpath << std::endl;
		if (aggpath_token > 0) {
			renderer_->removeDynamicLine(aggpath_token);
		}
		Eigen::MatrixXd adj;
		adj.resize(aggpath.size(), ND + 1);
		for (size_t i = 0 ; i < aggpath.size(); i++) {
			adj.row(i) = aggpath[i];
			adj(i, ND) = 2.0; // Note: 2D only
		}
		aggpath_token = renderer_->addDynamicLine(adj);
	}
#endif

	static void pause()
	{
		std::cerr << "Press enter to continue" << std::endl;
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
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
		std::set<int> agg_line_tokens_;
	};
#endif

	template<typename Node>
	static void visSplit(Node* node)
	{
		renderer_->addSplit(node->getMedian(), node->getMins(), node->getMaxs());
	}

	template<typename Node>
	static void visCertain(Node* node)
	{
		renderer_->addCertain(node->getMedian(),
				node->getMins(),
				node->getMaxs(),
				node->getState() == Node::kCubeFree);
	}

	static void setRenderer(NaiveRenderer* renderer) { renderer_ = renderer; }
private:
	static NaiveRenderer* renderer_;

	template<typename Node>
	static Eigen::MatrixXd build_line(Node* node, Node* neighbor)
	{
		Eigen::MatrixXd adj;
		adj.resize(2, ND + 1);
		adj.row(0) = node->getMedian();
		adj.row(1) = neighbor->getMedian();
		adj(0, ND) = 2.0;
		adj(1, ND) = 2.0;
		return adj;
	}

#if SHOW_AGGPATH
	static int aggpath_token;
#endif
};

#endif
