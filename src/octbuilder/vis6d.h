/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef VIS6D_H
#define VIS6D_H

#include <goct/nullvisualizer.h>
#include "naiverenderer.h"

class NaiveVisualizer6D : public NodeCounterVisualizer {
public:
	static void initialize()
	{
		NodeCounterVisualizer::initialize();
	}

	static void visAggPath(const std::vector<Eigen::VectorXd>& aggpath)
	{
		// std::cerr << "Aggressive path: " << aggpath << std::endl;
		if (aggpath_token > 0) {
			renderer_->removeDynamicLine(aggpath_token);
		}
		Eigen::MatrixXd adj;
		adj.resize(aggpath.size(), 3);
		for (size_t i = 0 ; i < aggpath.size(); i++) {
			adj.row(i) = aggpath[i].block<3,1>(0,0);
		}
		aggpath_token = renderer_->addDynamicLine(adj);
	}

	template<typename Node>
	static void visCertain(Node* node)
	{
#if 1 // Only visualize Full
		if (!node->atState(Node::kCubeFull))
			return ;
#endif
		if (!contain_no_rotation(node))
			return ;
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
	static int aggpath_token;

	static bool has_zero(double min, double max)
	{
		return min <= 0.0 && max >= 0.0;
	}

	template<typename Node>
	static bool contain_no_rotation(Node* node)
	{
		auto mins = node->getMins();
		auto maxs= node->getMaxs();
		return has_zero(mins(3), maxs(3))
		    && has_zero(mins(4), maxs(4))
		    && has_zero(mins(5), maxs(5));
	}
};

#endif
