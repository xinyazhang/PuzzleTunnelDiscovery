#include <unistd.h>
#include <string>
#include <unordered_map>
#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <igl/barycenter.h>
#include <igl/viewer/Viewer.h>
#include <igl/jet.h>
#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include <time.h>

using std::string;
using std::endl;
using std::cerr;
using std::fixed;
using std::vector;

void usage()
{
	cerr << "Arguments: <Robot> <Workspace>" << endl;
}

class Mink {
private:
	Eigen::MatrixXd RV_;
	Eigen::MatrixXi RF_;
	Eigen::MatrixXd WV_;
	Eigen::MatrixXi WF_;
	double init_radius = 2.0;
	double ratio_ = 1.0;
	static constexpr int nsubd_ = 32;

	void blend()
	{
		blend_vertices();
		blend_faces();
	}
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;

	void move_robot_to_center()
	{
		double x = RV_.col(0).minCoeff() + RV_.col(0).maxCoeff();
		double y = RV_.col(1).minCoeff() + RV_.col(1).maxCoeff();
		double z = RV_.col(2).minCoeff() + RV_.col(2).maxCoeff();
		x /= 2;
		y /= 2;
		z /= 2;
		RV_.col(0) = RV_.col(0).array() - x;
		RV_.col(1) = RV_.col(1).array() - y;
		RV_.col(2) = RV_.col(2).array() - z;
	}

	void blend_vertices()
	{
		V_.resize(RV_.rows() + WV_.rows(), RV_.cols());
		V_.block(0, 0, RV_.rows(), RV_.cols()) = RV_;
		V_.block(RV_.rows(), 0, WV_.rows(), RV_.cols()) = WV_;
	}

	void blend_faces()
	{
		F_.resize(RF_.rows() + WF_.rows(), RF_.cols());
		F_.block(0, 0, RF_.rows(), RF_.cols()) = RF_;
		F_.block(RF_.rows(), 0, WF_.rows(), RF_.cols()) = WF_.array() + RV_.rows();
	}
	Eigen::Vector3d robot_center_;

	void recalculate_square(Eigen::MatrixXd& V,
				double radius,
				int n)
	{
		for (int i = 0; i < 4; i++) {
			double theta = i * M_PI / 2.0;
			double dtheta = M_PI / 2.0 / n;
			Eigen::Vector3d center = V.row(i);
			for (int j = 0; j <= n; j++) {
				V.row(4 + i * (n+1) + j) = center + radius * Eigen::Vector3d(cos(theta + dtheta * j), sin(theta + dtheta * j), 0.0);
			}
		}
	}

	void build_square(Eigen::MatrixXd& V,
                          Eigen::MatrixXi& F,
                          double radius)
	{
		constexpr int n = nsubd_;
		V.resize(4 + (n+1) * 4, 3);
		F.resize(2 + n * 4 + 8, 3);
		//F.resize(2 + n * 4, 3);
		V.row(0) = Eigen::Vector3d( 1,  1, 0);
		V.row(1) = Eigen::Vector3d(-1,  1, 0);
		V.row(2) = Eigen::Vector3d(-1, -1, 0);
		V.row(3) = Eigen::Vector3d( 1, -1, 0);
		F.row(0) = Eigen::Vector3i(0, 1, 3);
		F.row(1) = Eigen::Vector3i(3, 1, 2);
		recalculate_square(V, radius, n);
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < n; j++) {
				F.row(2 + i * n + j) = Eigen::Vector3i(i, 4 + i * (n+1) + j, 4 + i * (n+1) + j + 1);
			}
		}
		F.row(2 + n * 4) = Eigen::Vector3i(0, 4 + n, 4 + n + 1);
		F.row(2 + n * 4 + 1) = Eigen::Vector3i(0, 4 + n + 1, 1);
		F.row(2 + n * 4 + 2) = Eigen::Vector3i(1, 4 + 2 * n + 1, 4 + 2 * n + 2);
		F.row(2 + n * 4 + 2 + 1) = Eigen::Vector3i(1, 4 + 2 * n + 2, 2);
		F.row(2 + n * 4 + 4) = Eigen::Vector3i(2, 4 + 3 * n + 2, 4 + 3 * n + 3);
		F.row(2 + n * 4 + 4 + 1) = Eigen::Vector3i(2, 4 + 3 * n + 3, 3);
		F.row(2 + n * 4 + 6) = Eigen::Vector3i(3, 4 + 4 * n + 3, 4);
		F.row(2 + n * 4 + 6 + 1) = Eigen::Vector3i(3, 4, 0);
	}

public:
	Mink(const string& robot,
	     const Eigen::Vector3d& robot_center
	     )
		:robot_center_(robot_center)
	{
		igl::readPLY(robot, RV_, RF_);
		build_square(WV_, WF_, init_radius * (1 - ratio_));
		//build_square(WV_, WF_, 0.5);
		//move_robot_to_center();
		blend_vertices();
		blend_faces();
	}

	void init_color(igl::viewer::Viewer& viewer)
	{
		viewer.data.set_mesh(V_, F_);
		viewer.data.set_face_based(false);

		Eigen::MatrixXd C;
		C.resize(V_.rows(), 3);
		for (int i = 0; i < RV_.rows(); i++)
			C.row(i) = Eigen::Vector3d(1.0, 0.0, 0.0);
		for (int i = RV_.rows(); i < V_.rows(); i++)
			C.row(i) = Eigen::Vector3d(0.0, 1.0, 0.0);
		viewer.data.set_colors(C);
	} 

	void save_frame()
	{
		// FIXME
	}

	void update_frame(igl::viewer::Viewer& viewer)
	{
		mink_scale();
		viewer.data.set_mesh(V_, F_);
		std::cerr << "updated to: " << ratio_ << std::endl;
	}

	bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
	{
		return false;
	}

	bool next_frame() 
	{
		ratio_ -= 0.00625;
		if (ratio_ < 0.0) {
			ratio_ = 0.0;
			return false;
		}
		return true;
	}

	void mink_scale()
	{
		Eigen::Transform<double, 3, Eigen::Affine> t;
		//t.scale(ratio_);
		//t.scale(2.0);
		//t.translate(robot_center_);
#pragma omp parallel for
		for (int i = 0; i < RV_.rows(); i++) {
			//Eigen::Vector4d vec = RV_.row(i);
			//vec(3) = 1.0;
			V_.row(i) = robot_center_;
			V_.row(i) += RV_.row(i) * (ratio_ * init_radius);
		}
		recalculate_square(WV_, init_radius * (1 - ratio_), nsubd_);
		V_.block(RV_.rows(), 0, WV_.rows(), RV_.cols()) = WV_;
	}
};

int main(int argc, char* argv[])
{
	string iprefix, ffn, pfn;
	if (argc < 2) {
		std::cerr << "Missing input file" << endl;
		usage();
		return -1;
	}

	igl::viewer::Viewer viewer;
	viewer.core.orthographic = true;
	Mink mink(argv[1], Eigen::Vector3d(4.0, 1.5, 0.0));
	mink.init_color(viewer);

	viewer.callback_key_pressed = [&mink](igl::viewer::Viewer& viewer, unsigned char key, int modifier) -> bool { return mink.key_down(viewer, key, modifier); } ;
	viewer.callback_pre_draw = [&mink](igl::viewer::Viewer& viewer) -> bool
	{
		if (viewer.core.is_animating) {
			if (mink.next_frame()) {
				mink.update_frame(viewer);
			}
		}
		return false;
	};
	viewer.core.is_animating = true;
	viewer.core.animation_max_fps = 10.;
	viewer.launch();

	return 0;
}
