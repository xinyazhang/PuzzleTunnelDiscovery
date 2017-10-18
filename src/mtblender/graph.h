#ifndef MTBLENDER_GRAPH_H
#define MTBLENDER_GRAPH_H

#include "config.h"
#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <utility>
#include <memory>
#include <istream>

struct Vertex {
	int index;
	Eigen::Matrix<double, kTrSpaceDim, 1> tr;
	Eigen::Quaternion<double> rot;

	Vertex(std::istream& fin, int index_counter);
};

using Edge = std::pair<int, int>;

std::istream& operator >> (std::istream& fin, Edge& e);
std::istream& operator >> (std::istream& fin, Vertex& v);

class Graph {
	struct GraphData;
public:
	Graph();
	~Graph();

	/* Load and merge accept temporary istream objects */
	void loadRoadMap(std::istream&&);
	void mergePath(std::istream&&);
	void printGraph(std::ostream&);
private:
	std::unique_ptr<GraphData> d_;
};

#endif
