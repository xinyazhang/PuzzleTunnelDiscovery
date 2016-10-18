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
	Eigen::MatrixXd RV_, initRV_;
	Eigen::MatrixXi RF_, initRF_;
	Eigen::MatrixXd WV_;
	Eigen::MatrixXi WF_;
	double t_ = 0.0;

	void blend()
	{
		blend_vertices();
		blend_faces();
	}
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;

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
	Eigen::Vector3d robot_handle_;

	void build_robot()
	{
		RV_.resize(3, 3);
		RV_.row(0) << -1, 0, 0;
		RV_.row(1) << -3, -1.5, 0;
		RV_.row(2) << -2.2, -2.4, 0;
		RF_.resize(1, 3);
		RF_.row(0) << 0, 1, 2;

		initRV_ = RV_;
		initRF_ = RF_;
	}
	void build_ws()
	{
		WV_.resize(3, 3);
		WV_.row(0) << 1.8, 3, 0;
		WV_.row(1) << 0.75, 0, 0;
		WV_.row(2) << 3, -1.25, 0;
		WF_.resize(1, 3);
		WF_.row(0) << 0, 1, 2;
	}
public:
	Mink()
	{
		build_robot();
		build_ws();
		robot_handle_ = RV_.row(0);

		blend_vertices();
		blend_faces();
	}

	void init_viewer(igl::viewer::Viewer& viewer)
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

	void update_frame(igl::viewer::Viewer& viewer)
	{
		blend_vertices();
		viewer.data.set_mesh(V_, F_);
	}

	bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
	{
		return false;
	}

	bool next_frame() 
	{
		t_ += 1.0/128.0;
		int it = int(t_);
		double ratio = 1.0 - (t_ - double(it));
		int vid0 = it % WV_.rows();
		int vid1 = (it + 1) % WV_.rows();
		Eigen::Vector3d v0 = WV_.row(WF_(0, vid0));
		Eigen::Vector3d v1 = WV_.row(WF_(0, vid1));
		Eigen::Vector3d handle = v0 * ratio + v1 * (1 - ratio);

		Eigen::Vector3d tr = handle - robot_handle_;
		for (int i = 0; i < RV_.rows(); i++) {
			RV_.row(i) = initRV_.row(i) + tr.transpose();
		}

		return true;
	}
};

int main(int argc, char* argv[])
{
	igl::viewer::Viewer viewer;
	viewer.core.orthographic = true;
	Mink mink;
	mink.init_viewer(viewer);
	viewer.core.clear_framebuffers();
	glClearColor(0.3f, 0.3f, 0.5f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	viewer.core.camera_eye << 0, 0, 10;

#if 1
	viewer.callback_key_pressed = [](igl::viewer::Viewer& viewer, unsigned char key, int modifier) -> bool
	{
		if (key == 'C') {
			glClearColor(0.3f, 0.3f, 0.5f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}
	};
#endif
	viewer.callback_pre_draw = [&mink](igl::viewer::Viewer& viewer) -> bool
	{
		if (viewer.core.is_animating) {
			if (mink.next_frame()) {
				mink.update_frame(viewer);
			}
		}
		return false;
	};
	viewer.core.is_animating = false;
	viewer.core.animation_max_fps = 60.;
	viewer.launch();

	return 0;
}
