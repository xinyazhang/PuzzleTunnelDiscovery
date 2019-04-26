#include <getopt.h>
#include <omp.h>
#include <strings.h>
#include <string>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <unsupported/Eigen/SparseExtra>
#include <igl/ray_mesh_intersect.h>
#include <igl/readOBJ.h>
#include <numerical/stability.h>

#include <tetio/readtet.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	std::cerr << "Options: -i <tetgen file prefix> [-o output_initial_vector -0 boundary_value -D -N <fp value> -p <obs file>] Boundary Description" << endl;
	std::cerr << "\t-D: enable Dirichlet condition on the boundary" << endl
		  << "\t-N: enable and set Neumann condition on the boundary" << endl
		  << "\t-p: provide the configuration space obstacle geometry" << endl
		  << "\tBoundary Description:" << endl
		  << "\t\tX- X+ Y- Y+ Z- Z+ : boundary faces at the XYZ axis" << endl
		  << "\t\tauto : auto detect free area boundaries." << endl
		  << "\t\t\tthis function requires -p option." << endl;
}

const int kCoords[][2] = {
	{0, -1}, // X-
	{0,  1}, // X+
	{1, -1}, // Y-
	{1,  1}, // Y+
	{2, -1}, // Z-
	{2,  1}, // Z+
};

const char* kCoordsName[] = {
	"X-",
        "X+",
        "Y-",
        "Y+",
        "Z-",
        "Z+"
};

enum BOUNDARY_CONDITION {
	BC_NONE,
	BC_DIRICHLET,
	BC_NEUMANN
};

void set_IV(Eigen::VectorXd& iv,
	    const Eigen::MatrixXd& V,
	    int coord,
	    double target_value,
	    double bv0
	    )
{
	for (int i = 0; i < V.rows(); i++) {
		double coord_value = V(i, coord);
		if (fpclose(target_value, coord_value))
			iv(i) = bv0;
	}
}

void markup_free_vertices(const Eigen::MatrixXd& ObV,
		const Eigen::MatrixXi& ObF,
		const Eigen::MatrixXd& V,
		Eigen::VectorXi &Vfree)
{
	Vfree.resize(V.rows());
	#pragma omp parallel for
	for (int i = 0; i < V.rows(); i++) {
		Eigen::Vector3d source = V.row(i);
		Eigen::Vector3d dest = source;
		source(2) = - M_PI;
		dest(2) = M_PI * 2;
		igl::Hit dontcare;
		bool intersect = igl::ray_mesh_intersect(source, dest, ObV, ObF, dontcare);
		Vfree(i) = intersect ? 0 : 1;
	}
}

void detect_boundary_vertices(const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& P,
		Eigen::VectorXi& Vfree)
{
	Eigen::VectorXi vb;
	vb.resize(Vfree.size());
	std::unordered_map<int, std::set<int>> vert_neighbors;
	for (int i = 0; i < P.rows(); i++) {
		for (int j = 0; j < P.cols(); j++) {
			int vert1 = P(i,j);
			for (int k = j + 1; k < P.cols(); k++) {
				int vert2 = P(i,k);
				vert_neighbors[vert1].insert(vert2);
				vert_neighbors[vert2].insert(vert1);
			}
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < V.rows(); i++) {
		if (Vfree(i) == 0)
			continue; // Skip non-free vertices
		bool boundary = false;
		for (int neigh : vert_neighbors[i]) {
			if (Vfree(neigh) == 0) {
				boundary = true;
				break;
			}
		}
		vb(i) = boundary ? 1 : 0;
	}

	Vfree.swap(vb);
}

void set_boundary_value(const Eigen::VectorXi& Vfree,
		Eigen::VectorXd& IV,
		double bv0)
{
	for (int i = 0; i < Vfree.rows(); i++) {
		if (Vfree(i))
			IV(i) = bv0;
	}
}

int main(int argc, char* argv[])
{
	int opt;
	string iprefix, ofn, obfn;
	double bv0 = 1.0;
	double nv0 = -1.0;
	BOUNDARY_CONDITION bc = BC_NONE;
	while ((opt = getopt(argc, argv, "i:o:0:DNp:")) != -1) {
		switch (opt) {
			case 'i': 
				iprefix = optarg;
				break;
			case 'o':
				ofn = optarg;
				break;
			case '0':
				bv0 = atof(optarg);
				break;
			case 'D':
				bc = BC_DIRICHLET;
				break;
			case 'p':
				obfn = optarg;
				break;
			case 'N':
				bc = BC_NEUMANN;
				nv0 = atof(optarg);
				break;
			default:
				std::cerr << "Unrecognized option: " << optarg << endl;
				usage();
				return -1;
		}
	}
	bool boundary_enabled[6] = {false, false, false, false, false, false};
	bool detect_boundary = false;
	for(int i = optind; i < argc; i++) {
		if (strcasecmp("auto", argv[i]) == 0) {
			detect_boundary = true;
		} else {
			for(int k = 0; k < 6; k++) {
				if (strcasecmp(kCoordsName[k], argv[i]) == 0) {
					boundary_enabled[k] = true;
				}
			}
		}
	}

	if (iprefix.empty()) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}
	Eigen::MatrixXd ObV;
	Eigen::MatrixXi ObF;
	if (detect_boundary) {
		if (obfn.empty()) {
			std::cerr << "Missing obstacle file (-p)" << endl;
			usage();
			return -1;
		} else {
			if (!igl::readOBJ(obfn, ObV, ObF)) {
				std::cerr << "Cannot read " << obfn << " as OBJ file" << endl;
				return -1;
			}
		}
	}
	if (ofn.empty()) {
		ofn = iprefix + ".Bcond";
	}

	Eigen::MatrixXd V;
	Eigen::MatrixXi E;
	Eigen::MatrixXi P;
	Eigen::VectorXi EBM;
	Eigen::MatrixXd IV;
	try {
		readtet(iprefix, V, E, P, &EBM);
		Eigen::VectorXd IV;
		IV.setZero(V.rows());
		Eigen::VectorXd minV, maxV;
		minV = V.colwise().minCoeff();
		maxV = V.colwise().maxCoeff();
		if (detect_boundary) {
			Eigen::VectorXi Vfree;
			markup_free_vertices(ObV, ObF, V, Vfree);
			detect_boundary_vertices(V, P, Vfree);
			set_boundary_value(Vfree, IV, bv0);
		}
		for(int k = 0; k < 6; k++) {
			if (!boundary_enabled[k])
				continue;
			int coord = kCoords[k][0];
			set_IV(IV,
				V,
				coord,
				kCoords[k][1] > 0 ? maxV(coord) : minV(coord),
				bv0
			      );
		}
		std::ofstream fout;
		fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
		fout.open(ofn);
		fout << IV.rows() << endl << IV << endl;
		fout.close();
		// FIXME: Support Neumann BC. (Dirichlet BC is supported by
		// heat directly.
	} catch (std::runtime_error& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}
	return 0;
}
