#ifndef OBJAUTOUV_MESH_H
#define OBJAUTOUV_MESH_H

#include <Eigen/Core>
#include <vector>

struct UVBox;

struct Mesh {
	Mesh();
	~Mesh();
	// Vertex attributes
	Eigen::MatrixXd V;
	Eigen::MatrixXd UV;
	Eigen::MatrixXd N;

	// Indices of face attributes
	Eigen::MatrixXi F;
	Eigen::MatrixXi FUV;
	Eigen::MatrixXi FN;

	// Intermediate data
	Eigen::MatrixXd el;             // Edge length
	Eigen::MatrixXd face_normals;
	Eigen::MatrixXi face_pair;      // # x 2 matrices for face pairs that formulates a box
	//// Triangle-Triangle Adjacency data
	//// Convention: faces/edges [1,2], [2,3], [0,1]
	Eigen::MatrixXi tta_face;       // adjacency face
	// Eigen::MatrixXi tta_edge;       // the connecting edge index from the cooresponding adj triangle
	Eigen::VectorXd face_longedge;
	Eigen::VectorXi face_longedge_id;   // (Relative) ID of longest edge in a face
	Eigen::VectorXi adjface_longedge;  // ID of adjacenct face

	// Intermediate routines
	void PairWithLongEdge();

	std::vector<UVBox> boxes;
	UVBox CreateBB(int f, int other_f = -1);

	// Final routine
	//   After calling PairWithLongEdge to set boxes,
	//   call lib/rectpack to pack these boxes.
	void Program();

	double rec_u_size, rec_v_size;
};

#endif
