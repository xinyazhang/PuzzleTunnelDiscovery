#include "graph.h"
#include <vector>
#include <cmath>
#include <ompl/datastructures/NearestNeighborsGNAT.h>

using std::unique_ptr;
using std::endl;

Vertex::Vertex(std::istream& fin, int index_counter)
	:index(index_counter)
{
	fin >> *this;
}

std::istream& operator >> (std::istream& fin, Edge& e)
{
	return fin >> e.first >> e.second;
}

std::istream& operator >> (std::istream& fin, Vertex& v)
{
	for (int i = 0; i < v.tr.rows(); i++) {
		fin >> v.tr(i);
	}
	/* OMPL uses xyzw convention */
	fin >> v.rot.x() >> v.rot.y() >> v.rot.z() >> v.rot.w();
	return fin;
}

std::ostream& operator << (std::ostream& fout, Vertex& v)
{
	const char* sep = "";
	for (int i = 0; i < v.tr.rows(); i++) {
		fout << sep << v.tr(i);
		sep = " ";
	}
	/* Our library uses wxyz convention */
	fout << v.rot.w() << " "
	     << v.rot.x() << " "
	     << v.rot.y() << " "
	     << v.rot.z();
	return fout;
}

struct Graph::GraphData {
	std::vector<unique_ptr<Vertex>> V_;
	std::vector<Edge> E_;
	std::vector<Edge> PotE_; // Potentially existing E
	int index_counter;
	std::unique_ptr<ompl::NearestNeighborsGNAT<Vertex*>> nn_;

	static double distance(const Vertex* plhv, const Vertex* prhv)
	{
		const Vertex& lhv = *plhv;
		const Vertex& rhv = *prhv;
		double trdist = (lhv.tr - rhv.tr).norm();
		double rotdist = std::abs(std::acos(lhv.rot.dot(rhv.rot)));
		return trdist + rotdist;
	}

	void buildNN()
	{
		nn_.reset(new ompl::NearestNeighborsGNAT<Vertex*>());
		nn_->setDistanceFunction(distance);
		for (const auto& vp : V_)
			nn_->add(vp.get());
	}

	void addRRTPath(const std::vector<Vertex*>& vlist)
	{
		/* Add path in RRT to edge set */
		for (size_t i = 1; i < vlist.size(); i++) {
			const auto curr = vlist[i-1];
			const auto next = vlist[i];
			E_.emplace_back(curr->index, next->index);
		}
		std::vector<Vertex*> neighbors;
		for (const auto v : vlist) {
			nn_->nearestK(v, kNearest, neighbors);
			for (const auto neigh : neighbors)
				PotE_.emplace_back(v->index, neigh->index);
			V_.emplace_back(v);
		}
	}
};

Graph::Graph()
	:d_(new GraphData)
{
}

Graph::~Graph()
{
}

void Graph::loadRoadMap(std::istream&& fin)
{
	d_->V_.clear();
	d_->E_.clear();
	d_->index_counter = 0;

	while (!fin.eof()) {
		char type;
		fin >> type;
		if (type == 'v') {
			d_->V_.emplace_back(new Vertex(fin, d_->index_counter));
			d_->index_counter++;
		} else if (type == 'e') {
			Edge e;
			fin >> e;
			d_->E_.emplace_back(e);
		}
	}

	d_->buildNN();
}

void Graph::mergePath(std::istream&& fin)
{
	std::vector<Vertex*> vlist;
	while (!fin.eof()) {
		/*
		 * FIXME: Break if reading failed?
		 */
		vlist.emplace_back(new Vertex(fin, d_->index_counter));
		d_->index_counter++;
	}
	d_->addRRTPath(vlist);
}

void Graph::printGraph(std::ostream& fout)
{
	fout.precision(17);
	for (const auto& v : d_->V_) {
		fout << "v " << *v << endl;
	}
	for (const auto& e: d_->E_) {
		fout << "e " << e.first << " " << e.second << endl;
	}
	for (const auto& e: d_->PotE_) {
		fout << "p " << e.first << " " << e.second << endl;
	}
}
