#include "readvoronoi.h"
#include "read.hpp"
#include <fstream>
#include <stdexcept>

using std::runtime_error;
	
extern int read_vertices(Eigen::MatrixXd& V, std::istream& fin);

std::istream& operator>>(std::istream& fin, VoronoiEdge& ve)
{
	skip_spaces_and_comments(fin);
	fin >> ve.edgeno >> ve.v0idx >> ve.v1idx;
	if (ve.v1idx < 0) {
		fin >> ve.ray.x() >> ve.ray.y() >> ve.ray.z();
	}
	return fin;
}

std::istream& operator>>(std::istream& fin, VoronoiFace& vf)
{
	skip_spaces_and_comments(fin);
	fin >> vf.faceno >> vf.cell0 >> vf.cell1;
	int nedge = read<int>(fin);
	vf.edgenos.clear();
	for(int i = 0; i < nedge; i++) {
		int edgeno = 0;
		fin >> edgeno;
		vf.edgenos.emplace_back(edgeno);
	}
	return fin;
}

std::istream& operator>>(std::istream& fin, VoronoiCell& vf)
{
	skip_spaces_and_comments(fin);
	int nface;
	fin >> vf.cellno >> nface;
	for(int i = 0; i < nface; i++) {
		int faceno;
		fin >> faceno;
		vf.facenos.emplace_back(faceno);
	}
	return fin;
}

// We can implement them as template functions, but we may need to add special
// functions afterwards, so...
void read_voronoi_edges(std::istream& fin, std::vector<VoronoiEdge>& vedges, int)
{
	int nedge = read<int>(fin);
	for(int i = 0; i < nedge; i++) {
		vedges.emplace_back(read<VoronoiEdge>(fin));
	}
}

void read_voronoi_faces(std::istream& fin, std::vector<VoronoiFace>& vfaces, int)
{
	int nface = read<int>(fin);
	for(int i = 0; i < nface; i++) {
		vfaces.emplace_back(read<VoronoiFace>(fin));
	}
}

void read_voronoi_cells(std::istream& fin, std::vector<VoronoiCell>& vcells, int)
{
	int ncell = read<int>(fin);
	for(int i = 0; i < ncell; i++) {
		vcells.emplace_back(read<VoronoiCell>(fin));
	}
}

int readvoronoi(const std::string& iprefix,
		Eigen::MatrixXd& vnodes,
		std::vector<VoronoiEdge>& vedges,
		std::vector<VoronoiFace>& vfaces,
		std::vector<VoronoiCell>& vcells)
{
	std::ifstream nodef(iprefix+".v.node");
	if (!nodef.is_open())
		throw runtime_error("Cannot open "+iprefix+".v.node for read");
	std::ifstream edgef(iprefix+".v.edge");
	if (!edgef.is_open())
		throw runtime_error("Cannot open "+iprefix+".v.edge for read");
	std::ifstream facef(iprefix+".v.face");
	if (!facef.is_open())
		throw runtime_error("Cannot open "+iprefix+".v.face for read");
	std::ifstream cellf(iprefix+".v.cell");
	if (!cellf.is_open())
		throw runtime_error("Cannot open "+iprefix+".v.cell for read");

	int base = read_vertices(vnodes, nodef);
	read_voronoi_edges(edgef, vedges, base);
	read_voronoi_faces(facef, vfaces, base);
	read_voronoi_cells(cellf, vcells, base);
	return base;
}
