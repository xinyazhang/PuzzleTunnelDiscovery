/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef TEXT_VISUALIZER_H
#define TEXT_VISUALIZER_H

#include <goct/nullvisualizer.h>
#include <iostream>

class TextVisualizer : public NodeCounterVisualizer {
public:
	static void initialize()
	{
		NodeCounterVisualizer::initialize();
	}

	template<typename Node>
	static void visSplit(Node* node)
	{
#if 0
		std::cerr << "Split node: " << *node << std::endl;
#endif
		NodeCounterVisualizer::visSplit(node);
	}

#if 0
	template<typename Node>
	static void visCertain(Node* node)
	{
		if (node->atState(Node::kCubeFree))
			std::cerr << "Clear cube: " << *node << std::endl;
		else
			std::cerr << "Solid cube: " << *node << std::endl;
		auto median = node->getMedian();
		if (fabs(median(0)) < 0.25 &&
		    fabs(median(1)) < 0.25 &&
		    median(2) > 0
		    ) {
			std::cerr << "Clear Z+ Axis cube: " << *node << std::endl;
		}
	}
#endif
#if 0
	template<typename Node>
	static void visPending(Node* node)
	{
		std::cerr << "Add to list: " << *node << std::endl;
	}

	template<typename Node>
	static void visPop(Node* node)
	{
		std::cerr << "Pop from list: " << *node << std::endl;
	}
#endif
#if 0
	template<typename Node>
	static void visAdj(Node* node, Node* neighbor)
	{
		std::cerr << "Adjacency:\n\t" << *node << "\n\t" << *neighbor << std::endl;
	}

	template<typename Node>
	static void visAggAdj(Node* node, Node* neighbor)
	{
		std::cerr << "Adjacency (aggressive):\n\t" << *node << "\n\t" << *neighbor << std::endl;
	}
#endif
};

#endif
