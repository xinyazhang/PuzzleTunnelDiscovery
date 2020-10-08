/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <unistd.h>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>
#include <string>
#include <set>
#include <memory>
#include <unsupported/Eigen/SparseExtra>
//#include <boost/progress.hpp>
#include <tetio/readtet.h>
#include <tetio/writetet.h>
#include <igl/writePLY.h>
//#include <Eigen/SparseLU> 
//#include <Eigen/SparseCholesky>
//#include <Eigen/CholmodSupport>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> -o <output file prefix> VERTICES TO PICKUP" << endl;
	std::cerr << "\tNote: only V,E,P are printted" << endl;
}

void mark(const Eigen::MatrixXi& P, // P for Primitives, can be tetrahedra or edges
	const std::set<int>& pickset,
	Eigen::VectorXi& markP,
	Eigen::VectorXi& markV,
	int required_present = 1
	)
{
	markP.setZero(P.rows());
	for (int i = 0; i < P.rows(); i++) {
		int npresent = 0;
		for(int j = 0; j < P.cols(); j++) {
			if (pickset.find(P(i,j)) != pickset.end()) {
				npresent++;
				if (npresent >= required_present)
					break;
			}
		}
		if (npresent >= required_present) {
			markP(i) = 1;
			for(int j = 0; j < P.cols(); j++)
				markV(P(i,j)) = 1;
		}
	}
}

template<typename EigenMatrix>
void compact(const EigenMatrix& in,
	     const Eigen::VectorXi& mark,
	     EigenMatrix& out,
	     std::map<int, int>& old2new)
{
	int sum = mark.sum(); // Total number of elements to preserve
	out.resize(sum, in.cols());
	old2new.clear();
	for(int i = 0; i < in.rows(); i++) {
		if (mark(i) == 0)
			continue;
		int newidx = int(old2new.size());
		out.row(newidx) = in.row(i);
		old2new[i] = newidx;
		//std::cerr << i << "--->" << newidx << std::endl;
	}
}

void renumber(const std::map<int, int>& old2new,
	      Eigen::MatrixXi& io)
{
	for(int i = 0; i < io.rows(); i++) {
		for(int j = 0; j < io.cols(); j++) {
			auto iter = old2new.find(io(i,j));
			if (iter == old2new.end())
				throw std::runtime_error("old2new failed");
			io(i,j) = iter->second;
		}
	}
}

void ring1pick( const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& E,
		const Eigen::MatrixXi& P,
		const Eigen::MatrixXi& F,
		const std::vector<int>& picklist,
		bool edge_based,
		Eigen::MatrixXd& OV,
		Eigen::MatrixXi& OE,
		Eigen::MatrixXi& OP,
		Eigen::MatrixXi& OF,
		Eigen::VectorXi& FBM)
{
	using namespace Eigen;
	VectorXi markV;
	VectorXi markP;
	VectorXi markE;
	VectorXi markF;
	std::map<int, int> old2new, dontcare;
	std::set<int> pickset;

	markV.setZero(V.rows());
	for(auto pick : picklist) {
		pickset.emplace(pick);
	}
	int npresent = 1;
	if (edge_based)
		npresent = 2; // Edge based pick: two vertices present in Tet

	mark(P, pickset, markP, markV, npresent);
	std::set<int> pickset_from_tet; // selected tets -> selected vertices
	for(int i = 0; i < P.rows(); i++) {
		if (markP(i) == 0)
			continue;
		pickset_from_tet.emplace(P(i,0));
		pickset_from_tet.emplace(P(i,1));
		pickset_from_tet.emplace(P(i,2));
		pickset_from_tet.emplace(P(i,3));
	}
	mark(E, pickset_from_tet, markE, markV, 2); // Edges should always be in selected tets
	mark(F, pickset_from_tet, markF, markV, 3); // Faces' all vertices should all be in pickset_for_face

	compact(V, markV, OV, old2new);
	compact(E, markE, OE, dontcare);
	compact(P, markP, OP, dontcare);
	compact(F, markF, OF, dontcare);

	renumber(old2new, OE);
	renumber(old2new, OP);
	renumber(old2new, OF);

	std::set<int> pickset_new;
	for(auto pick : picklist) {
		int newpick = old2new[pick];
		pickset_new.emplace(newpick);
		//std::cerr << "new pickset " << newpick << endl;
	}
	FBM.resize(OF.rows());
	for(int i = 0; i < OF.rows(); i++) {
		FBM(i) = 1;
		for(int j = 0; j < OF.cols(); j++) {
			//std::cerr << "probe " << OF(i,j) << endl;
			if (pickset_new.find(OF(i,j)) != pickset_new.end()) {
				FBM(i) = 0;
			}
		}
	}
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int opt;
	string igf, ogf; // Input/Output Geometry Filename
	bool edge_based = false;
	while ((opt = getopt(argc, argv, "i:o:e")) != -1) {
		switch (opt) {
			case 'i':
				igf = optarg;
				break;
			case 'o':
				ogf = optarg;
				break;
			case 'e':
				edge_based = true;
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	std::vector<int> picklist;
	for(int i = optind; i < argc; i++) {
		picklist.emplace_back(atoi(argv[i]));
	}
	// Laplacian
	if (ogf.empty()) {
		std::cerr << "Missing output file (-l)" << endl;
		usage();
		return -1;
	}

	if (igf.empty()) {
		std::cerr << "Missing Geometry from tetgen output (-i)" << endl;
		usage();
		return -1;
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::MatrixXi F;
	Eigen::VectorXi EBM, FBM;
	try {
		readtet(igf, V, E, P, &EBM);
		readtet_face(igf, F, &FBM); // We don't need FBM actually
		Eigen::MatrixXd OV;
		Eigen::MatrixXi OE;
		Eigen::MatrixXi OP;
		Eigen::MatrixXi OF;
		ring1pick(V, E, P, F, picklist,
			  edge_based,
			  OV, OE, OP, OF, FBM);
		writetet(ogf, OV, OE, OP);
		writetet_face(ogf, OF, &FBM);
		igl::writePLY(ogf+".ply", OV, OF);
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	return 0;
}
