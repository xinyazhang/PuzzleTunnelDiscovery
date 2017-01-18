#include "nullvisualizer.h"
#include <iostream>

time_t NullVisualizer::last_time_;
unsigned long NodeCounterVisualizer::nsplit_nodes = 0;
std::vector<unsigned long> NodeCounterVisualizer::nsplit_histgram;

void NodeCounterVisualizer::periodicalReport()
{
	std::cerr << nsplit_nodes << " nodes splitted\t";
}

void NodeCounterVisualizer::showHistogram()
{
	std::cerr << "Number of splitted nodes per depth\n";
	for (size_t i = 0; i < nsplit_histgram.size(); i++) {
		std::cerr << "\tDepth: " << i
		          << "\t# of nodes\t" << nsplit_histgram[i]
		          << std::endl;
	}
}
