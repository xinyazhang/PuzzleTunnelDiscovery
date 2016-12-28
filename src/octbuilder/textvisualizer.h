#ifndef TEXT_VISUALIZER_H
#define TEXT_VISUALIZER_H

#include "nullvisualizer.h"
#include <iostream>

class TextVisualizer : public NullVisualizer {
public:
	template<typename Node>
	static void visSplit(Node* node)
	{
		std::cerr << "Split node: " << *node << std::endl;
	}
};

#endif
