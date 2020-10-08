/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/writeOBJ.h>
#include <igl/writePLY.h>
#include <igl/MeshBooleanType.h>
//#define IGL_NO_CORK
//#undef IGL_STATIC_LIBRARY
#include <getopt.h>
#include <meshbool/join.h>

#include <iostream>
#include <limits>

Eigen::MatrixXd VC;
Eigen::VectorXi J,I;
Eigen::MatrixXi FC;
igl::MeshBooleanType boolean_type(igl::MESH_BOOLEAN_TYPE_UNION);

const char * MESH_BOOLEAN_TYPE_NAMES[] =
{
  "Union",
  "Intersect",
  "Minus",
  "XOR",
  "Resolve",
};

#if 0
void update(igl::viewer::Viewer &viewer)
{
  //igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC,J);
  Eigen::MatrixXd C(FC.rows(),3);
  for(size_t f = 0;f<C.rows();f++)
  {
      C.row(f) = Eigen::RowVector3d(1,0,0);
#if 0
    if(J(f)<FA.rows())
    {
      C.row(f) = Eigen::RowVector3d(1,0,0);
    }else
    {
      C.row(f) = Eigen::RowVector3d(0,1,0);
    }
#endif
  }
  viewer.data.clear();
  viewer.data.set_mesh(VC,FC);
  viewer.data.set_colors(C);
}

bool key_down(igl::viewer::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    default:
      return false;
    case '.':
      boolean_type =
        static_cast<igl::MeshBooleanType>(
          (boolean_type+1)% igl::NUM_MESH_BOOLEAN_TYPES);
      break;
    case ',':
      boolean_type =
        static_cast<igl::MeshBooleanType>(
          (boolean_type+igl::NUM_MESH_BOOLEAN_TYPES-1)%
          igl::NUM_MESH_BOOLEAN_TYPES);
      break;
    case '[':
      viewer.core.camera_dnear -= 0.1;
      return true;
    case ']':
      viewer.core.camera_dnear += 0.1;
      return true;
  }
  //std::cout<<"A "<<MESH_BOOLEAN_TYPE_NAMES[boolean_type]<<" B."<<std::endl;
  //igl::copyleft::cgal::mesh_boolean(VA,FA,VB,FB,boolean_type,VC,FC);
  update(viewer);
  return true;
}
#endif

void build_cube(
		Eigen::MatrixXd& cubev,
		Eigen::MatrixXi& cubei,
		double xmin,
		double xmax,
		double ymin,
		double ymax,
		double zmin,
		double zmax
		)
{
	cubev.resize(8, 3);
	cubev.row(0) << xmin, ymin, zmin;
	cubev.row(1) << xmax, ymin, zmin;
	cubev.row(2) << xmax, ymax, zmin;
	cubev.row(3) << xmin, ymax, zmin;
	cubev.row(4) << xmin, ymin, zmax;
	cubev.row(5) << xmax, ymin, zmax;
	cubev.row(6) << xmax, ymax, zmax;
	cubev.row(7) << xmin, ymax, zmax;
#if 0
	cubei.resize(6,4);
	cubei.row(0) << 3,2,1,0;
	cubei.row(1) << 0,1,5,4;
	cubei.row(2) << 1,2,6,5;
	cubei.row(3) << 2,3,7,6;
	cubei.row(4) << 0,4,7,3;
	cubei.row(5) << 4,5,6,7;
#else
	cubei.resize(12,3);
	cubei.row(0) << 3,2,1;
	cubei.row(1) << 1,0,3;
	cubei.row(2) << 1,2,5;
	cubei.row(3) << 2,6,5;
	cubei.row(4) << 4,5,7;
	cubei.row(5) << 5,6,7;
	cubei.row(6) << 0,1,4;
	cubei.row(7) << 1,5,4;
	cubei.row(8) << 2,3,7;
	cubei.row(9) << 2,7,6;
	cubei.row(10)<< 0,4,7;
	cubei.row(11)<< 0,7,3;
#endif
}

bool has_suffix(const std::string &str, const std::string &suffix)
{
	return str.size() >= suffix.size() &&
		str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
enum {
	OPT_PREFIX,
	OPT_NOINTERMEDIATE,
};

static struct option opts[] = {
	{"prefix", required_argument, NULL, OPT_PREFIX},
	{"NI", no_argument, NULL, OPT_NOINTERMEDIATE},
	{0, 0, 0, 0},
};

int main(int argc, char *argv[])
{
#if 1
	std::string prefix("blend-");
	int o;
	bool no_intermediate = false;
	do {
		o = getopt_long_only(argc, argv, "", opts, NULL);
		switch (o) {
			case OPT_PREFIX:
				prefix = optarg;
				break;
			case OPT_NOINTERMEDIATE:
				no_intermediate = false;
				break;
		};
	} while (o > 0);

	int start = optind;
	int nfile = argc - start;

	if (nfile < 2) {
		std::cerr << "Need at least two mesh files" << std::endl;
		return -1;
	}

	using namespace Eigen;
	using namespace std;
	std::vector<Eigen::MatrixXd> VArray(nfile);
	std::vector<Eigen::MatrixXi> FArray(nfile);
	std::cerr << "Loading " << argc << " meshes...";
	
	for(int i = start; i < argc; i++) {
		if (has_suffix(argv[i], ".obj"))
			igl::readOBJ(argv[i], VArray[i-start], FArray[i-start]);
		else if (has_suffix(argv[i], ".ply"))
			igl::readPLY(argv[i], VArray[i-start], FArray[i-start]);
		else {
			std::cerr << "Unrecognized file " << argv[i] << std::endl;
			continue;
		}
		std::cerr << "  " << i <<"(f: " << FArray[i-start].rows() <<") ";
	}
	std::cerr << " Done" << std::endl;;
	double xmax = VArray[0].col(0).maxCoeff();
	double xmin = VArray[0].col(0).minCoeff();
	double ymax = VArray[0].col(1).maxCoeff();
	double ymin = VArray[0].col(1).minCoeff();
	for(int i = 1; i < nfile; i++) {
		xmax = std::max(xmax, VArray[i].col(0).maxCoeff());
		xmin = std::min(xmin, VArray[i].col(0).minCoeff());
		ymax = std::max(ymax, VArray[i].col(1).maxCoeff());
		ymin = std::min(ymin, VArray[i].col(1).minCoeff());
	}
	Eigen::MatrixXd cubev;
	Eigen::MatrixXi cubei;
	build_cube(cubev, cubei, xmin - 0.25, xmax + 0.25, ymin - 0.25, ymax + 0.25, -0.25, 2 * M_PI + 0.25);
#endif
#if 0
	Eigen::MatrixXd cubev1;
	Eigen::MatrixXi cubei1;
	build_cube(cubev1, cubei1, -1, 1, -1, 1, -1, 1);
	std::cout << cubev1 << std::endl;
	std::cout << cubei1 << std::endl;
	std::cout << cubei1 << std::endl;
	igl::writeOBJ("cube1.obj", cubev1, cubei1);

	Eigen::MatrixXd cubev2;
	Eigen::MatrixXi cubei2;
	build_cube(cubev2, cubei2, 0, 2, 0, 2, 0, 2);
	std::cout << cubev2 << std::endl;
	std::cout << cubei2 << std::endl;
	igl::writeOBJ("cube2.obj", cubev2, cubei2);

	mesh_bool(
		cubev1, cubei1,
		cubev2, cubei2,
		igl::MESH_BOOLEAN_TYPE_MINUS,
		VC, FC);
	igl::writeOBJ("cubemin.obj", VC, FC);

	return 0;
#endif
#if 1
	VC = VArray[0];
	FC = FArray[0];
	std::cerr << "Mesh unoin... " ;
	std::cerr << "  " << 0 <<"(f: " << FC.rows() <<") ";
	for(size_t i = 1; i < VArray.size(); i++) {
		Eigen::MatrixXd RV;
		Eigen::MatrixXi RF;
		try {
			mesh_bool(
					VC,FC,
					VArray[i], FArray[i],
					igl::MESH_BOOLEAN_TYPE_UNION,
					RV,RF);
			VC.noalias() = RV;
			FC.noalias() = RF;
		} catch (...) {
			std::cerr << "  (" << i << ")";
		}

		if (!no_intermediate) {
			igl::writeOBJ(prefix+std::to_string(i)+".obj", VC, FC);
			//igl::writePLY(prefix+std::to_string(i)+".ply", VC, FC);
		}
		std::cerr << "  " << i <<"(f: " << FC.rows() <<") ";
	}
#else
	VC = cubev;
	FC = cubei;
	std::cerr << "Mesh subtraction... " ;
	std::cerr << "  " << 0 <<"(f: " << FC.rows() <<") ";
	for(size_t i = 0; i < VArray.size(); i++) {
		Eigen::MatrixXd RV;
		Eigen::MatrixXi RF;
		mesh_bool(
			VC,FC,
			VArray[i], FArray[i],
			igl::MESH_BOOLEAN_TYPE_MINUS,
			RV,RF);
		VC.noalias() = RV;
		FC.noalias() = RF;
		igl::writeOBJ("blend"+std::to_string(i)+".obj", VC, FC);
		std::cerr << "  " << i <<"(f: " << FC.rows() <<") ";
	}
#endif
	std::cerr << " Done" << std::endl;;
	std::cout.precision(std::numeric_limits<double>::max_digits10);
#if 0
	for(int i = 1293; i < 1296; i++) {
		std::cout << i << ": " << std::fixed << VC(i-1,2) << endl;
	}
#endif
	igl::writeOBJ(prefix+"done.obj", VC, FC);
	//igl::writePLY(prefix+"done.ply", VC, FC);
#if 0
	// Plot the mesh with pseudocolors
	igl::viewer::Viewer viewer;

	// Initialize
	update(viewer);

	viewer.core.show_lines = true;
	viewer.callback_key_down = &key_down;
	viewer.core.camera_dnear = 3.9;
	cout<<
		"Press '.' to switch to next boolean operation type."<<endl<<
		"Press ',' to switch to previous boolean operation type."<<endl<<
		"Press ']' to push near cutting plane away from camera."<<endl<<
		"Press '[' to pull near cutting plane closer to camera."<<endl<<
		"Hint: investigate _inside_ the model to see orientation changes."<<endl;
	viewer.launch();
#endif
	return 0;
}
