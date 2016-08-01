#include "readtet.h"
#include <ctype.h>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iostream> // For std::cerr
#include <fstream>
#include <vector>
#include "read.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;
using std::string;
using std::vector;
using std::runtime_error;
using stream = std::numeric_limits<std::streamsize>;

struct node {
	int idx;
	double x, y, z;
};

std::istream& operator>>(std::istream& fin, node& n)
{
	skip_spaces_and_comments(fin);
	fin >> n.idx >> n.x >> n.y >> n.z;
	return fin;
}

void skip_spaces_and_comments(std::istream& fin)
{
	char c;
	while (true) {
		do {
			c = fin.get();
		} while (isspace(c));
		if (c == '#')
			fin.ignore(stream::max(), '\n');
		else {
			fin.putback(c);
			break;
		}
	}
}

int read_vertices(MatrixXd& V, std::istream& fin)
{
	int npoint = read<int>(fin);
	int ndim = read<int>(fin);
	int nattr = read<int>(fin);
	int nbm = read<int>(fin);
	//cerr << npoint << ' ' << ndim << ' ' << nattr << ' ' << nbm << endl;
	if (ndim != 3)
		throw runtime_error("Unsupported .node file: #dim must be 3");

	vector<node> nodes(npoint);
	// The spec doesn't guarntee the nodes must be in-order.
	// So we have to read them all and sort.
	for(int i = 0; i < npoint; i++) {
		nodes[i] = read<node>(fin);
		// Drop attributes and boundary markers.
		for(int j = 0; j < nattr + nbm; j++)
			(void)read<string>(fin);
	}
	std::sort(nodes.begin(),
		  nodes.end(),
		  [] (const node& lhs, const node& rhs) {
			return lhs.idx < rhs.idx;
		  }
		 );
	if (nodes.empty())
		throw runtime_error("Invalid .node file: no nodes were read");
	
	V.resize(npoint, 3);
	for(int i = 0; i < npoint; i++) {
		V(i, 0) = nodes[i].x;
		V(i, 1) = nodes[i].y;
		V(i, 2) = nodes[i].z;
	}
	return nodes.front().idx;
}

void read_edges(std::istream& fin, MatrixXi& E, VectorXi* EBM, int base)
{
	int nedge = read<int>(fin);
	int bm = read<int>(fin);
	fin.ignore(stream::max(), '\n'); // Goto next line
	E.resize(nedge, 2);
	if (EBM)
		EBM->resize(nedge);
	else if (bm)
		throw runtime_error("readtet error: you must provide EBM vector for .edge files with boundary markers");
	int rebase = 0 - base;
	for(int i = 0; i < nedge; i++) {
		(void)read<int>(fin); // Skip the edge index
		E(i, 0) = read<int>(fin) + rebase;
		E(i, 1) = read<int>(fin) + rebase;
		if (bm) {
			(*EBM)(i) = read<int>(fin);
		}
		fin.ignore(stream::max(), '\n'); // Goto next line
	}
}

void read_tetrahedron(MatrixXi& P, std::istream& fin, int base)
{
	int ntetra = read<int>(fin);
	int nnode = read<int>(fin);
	int nattr = read<int>(fin);
	P.resize(ntetra, nnode);

	int rebase = 0 - base;
	for(int i = 0; i < ntetra; i++) {
		(void)read<int>(fin); // Skip the ele index

		for(int j = 0; j < nnode; j++)
			P(i, j) = read<int>(fin) + rebase; // Rebase to zero

		for(int j = 0; j < nattr; j++)
			(void)read<string>(fin); // skip additional attributes.
	}
}

void readtet(const string& iprefix, MatrixXd& V, MatrixXi& E, MatrixXi& P, VectorXi* EBMarker)
{
	int base = readtet(iprefix, V, P);
	std::ifstream edgef(iprefix+".edge");
	if (!edgef.is_open())
		throw runtime_error("Cannot open "+iprefix+".edge for read");
	read_edges(edgef, E, EBMarker, base);
}

int readtet(const std::string& iprefix,
	     Eigen::MatrixXd& V,
	     Eigen::MatrixXi& P)
{
	std::ifstream nodef(iprefix+".node");
	if (!nodef.is_open())
		throw runtime_error("Cannot open "+iprefix+".node for read");
	std::ifstream elef(iprefix+".ele");
	if (!elef.is_open())
		throw runtime_error("Cannot open "+iprefix+".ele for read");

	int base = read_vertices(V, nodef);
	read_tetrahedron(P, elef, base);
	return base;
}
