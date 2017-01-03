#ifndef NULL_VISUALIZER_H
#define NULL_VISUALIZER_H

#include <time.h>
#include <vector>
#include <Eigen/Core>
#include <vector>

/*
 * NullVisualizer: concept of Visualizer used by GOctreePathBuilder
 */

class NaiveRenderer;
class NullVisualizer {
public:
	static void initialize() { last_time_ = ::time(NULL); }
	template<typename Node>
	static void visAdj(Node*, Node* ) {}
	template<typename Node>
	static void visPathSegment(Node*, Node* ) {}

	template<typename Node>
	static void visAggAdj(Node*, Node* ) {}
	template<typename Node>
	static void withdrawAggAdj(Node*) {}

	template<typename Node>
	static void visSplit(Node*) {}
	template<typename Node>
	static void visCertain(Node*) {}
	template<typename Node>
	static void visPending(Node*) {}
	template<typename Node>
	static void visPop(Node*) {}

	static void visAggPath(const std::vector<Eigen::VectorXd>&) {}

	template<typename Node>
	static void trackFurestCube(Node* cube, Node* init_cube) {}

	static bool timerAlarming() { return ::time(NULL) > last_time_; }
	static void periodicalReport() {}
	static void rearmTimer() { last_time_ = ::time(NULL); }
	static void pause() {}

	struct Attribute {
	};

	static void setRenderer(NaiveRenderer*) {}
protected:
	static time_t last_time_;
};

class NodeCounterVisualizer : public NullVisualizer {
protected:
	static unsigned long nsplit_nodes;
	static std::vector<unsigned long> nsplit_histgram;

public:
	static void initialize()
	{
		nsplit_nodes = 0;
	}

	template<typename Node>
	static void visSplit(Node* node)
	{
		++nsplit_nodes;
		auto depth = node->getDepth();
		if (depth >= nsplit_histgram.size())
			nsplit_histgram.resize(depth + 1);
		++nsplit_histgram[depth];
	}

	static void periodicalReport();
	static void showHistogram();
};

#endif
