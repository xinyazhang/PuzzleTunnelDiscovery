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
	Eigen::MatrixXd RV_, initRV_, negrotRV_;
	Eigen::MatrixXi RF_, initRF_;
	Eigen::MatrixXd WV_;
	Eigen::MatrixXi WF_;
	double t_ = 0.0;
	double theta_ = 0.0;

	Eigen::MatrixXd V_;
	Eigen::MatrixXd C_;
	Eigen::MatrixXi F_;

	void do_blend_vertices(std::vector<Eigen::MatrixXd> mats,
			std::vector<Eigen::Vector3d> colors)
	{
		int nrows = 0;
		for(const auto& m : mats) {
			nrows += m.rows();
		}
		V_.resize(nrows, mats.front().cols());
		C_.resize(nrows, 3);
		int rowiter = 0;
		int coloriter = 0;
		for (const auto& m : mats) {
			V_.block(rowiter, 0, m.rows(), m.cols()) = m;
			for (int i = rowiter; i < rowiter + m.rows(); i++) {
				C_.row(i) = colors[coloriter];
			}
			rowiter += m.rows();
			coloriter++;
		}
	}

	void do_blend_faces(std::vector<Eigen::MatrixXi> mats,
			std::vector<Eigen::MatrixXd> vmats)
	{
		int nrows = 0;
		for(const auto& m : mats) {
			nrows += m.rows();
		}
		F_.resize(nrows, mats.front().cols());
		int rowiter = 0;
		int vrows = 0;
		int viter = 0;
		for (auto m : mats) {
			F_.block(rowiter, 0, m.rows(), m.cols()) = m.array() + vrows;
			//std::cerr << rowiter << std::endl;
			//std::cerr << m << std::endl;
			rowiter += m.rows();
			vrows += vmats[viter].rows();
			viter++;
		}
	}

	void blend()
	{
		Eigen::MatrixXd dispInitRV = initRV_;
		for (int i = 0; i < dispInitRV.rows(); i++)
			dispInitRV.row(i) -= Eigen::Vector3d(2.0,0.0,0.0);
		do_blend_vertices({RV_, WV_, negrotRV_, dispInitRV},
				{ Eigen::Vector3d(1.0, 0.0, 0.0),
				  Eigen::Vector3d(0.0, 1.0, 0.0),
				  Eigen::Vector3d(1.0, 0.0, 0.0),
				  Eigen::Vector3d(1.0, 1.0, 0.0),
				}
				);
		//do_blend_vertices({RV_, WV_}, {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0)});
		//do_blend_faces({RF_, WF_}, {RV_, WV_});
		do_blend_faces({RF_, WF_, RF_, RF_},
			{RV_, WV_, negrotRV_, dispInitRV}
			);
		
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

	void pickup_handle()
	{
		Eigen::Vector3d centroid(0.0, 0.0, 0.0);
		for (int i = 0; i < initRV_.rows(); i++)
			centroid += initRV_.row(i);
		centroid /= initRV_.rows();
		robot_handle_ = centroid;
	}

	void rebuild_basic_mink()
	{
		double theta = theta_;

		Eigen::Matrix3d rot;
		rot << cos(theta), -sin(theta), 0.0,
		       sin(theta), cos(theta), 0.0,
		       0.0, 0.0, 1.0;
		negrotRV_ = initRV_;
		for (int i = 0; i < RV_.rows(); i++) {
			negrotRV_.row(i) = rot * (initRV_.row(i).transpose() -
					robot_handle_);
		}
		for (int i = 0; i < RV_.rows(); i++) {
			negrotRV_.row(i) = robot_handle_ + (- negrotRV_.row(i).transpose());
		}
	}
public:
	Mink()
	{
		build_robot();
		build_ws();
		pickup_handle();

		theta_ = 0.0;
		rebuild_basic_mink();

		blend();
		//std::cerr << V_ << std::endl << F_;
	}

	void init_viewer(igl::viewer::Viewer& viewer)
	{
		viewer.data.set_mesh(V_, F_);
		viewer.data.set_face_based(false);

		viewer.data.set_colors(C_);
	} 

	void update_frame(igl::viewer::Viewer& viewer)
	{
		blend();
		viewer.data.set_mesh(V_, F_);
	}

	bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
	{
		return false;
	}

	bool next_frame() 
	{
		t_ += 1.0/60.0;
		int it = int(t_);
		double ratio = 1.0 - (t_ - double(it));
		int vid0 = it % WV_.rows();
		int vid1 = (it + 1) % WV_.rows();
		Eigen::Vector3d v0 = WV_.row(WF_(0, vid0));
		Eigen::Vector3d v1 = WV_.row(WF_(0, vid1));
		Eigen::Vector3d handle = v0 * ratio + v1 * (1 - ratio);

		Eigen::Vector3d tr = handle - robot_handle_;
		for (int i = 0; i < RV_.rows(); i++) {
			//std::cerr << "negrotRV_: " << i << std::endl;
			RV_.row(i) = negrotRV_.row(i) + tr.transpose();
		}

		return true;
	}

	void rotate(double direction)
	{
		theta_ += direction / 8.0 / M_PI;
		t_ = 0.0;
		//calc_rotation(theta_);
		rebuild_basic_mink();
	}

	void calc_rotation(double theta)
	{
	}
};

int main(int argc, char* argv[])
{
	igl::viewer::Viewer viewer;
	viewer.launch_init(true, false);
	viewer.core.orthographic = true;
	Mink mink;
	mink.init_viewer(viewer);
	viewer.core.clear_framebuffers();
	viewer.core.camera_eye << 0, 0, 10;
	viewer.core.clear_bits = GL_DEPTH_BUFFER_BIT;

#if 1
	viewer.callback_key_up = [&mink](igl::viewer::Viewer& viewer, unsigned int key, int modifier) -> bool {
		bool do_clear = false;
		//std::cerr << "Key " << key << " pressed" << std::endl;
		if (key == 'C' || key == 'c') {
			do_clear = true;
		} else if (key == GLFW_KEY_LEFT) {
			do_clear = true;
			mink.rotate(-1.0);
			mink.update_frame(viewer);
			//std::cerr << "Left pressed" << std::endl;
		} else if (key == GLFW_KEY_RIGHT) {
			do_clear = true;
			mink.rotate(1.0);
			mink.update_frame(viewer);
		}
		if (do_clear) {
			glClearColor(0.3f, 0.3f, 0.5f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}
		return false;
	};
#endif
	viewer.callback_pre_draw = [&mink](igl::viewer::Viewer& viewer) -> bool
	{
		if (viewer.core.is_animating) {
			if (mink.next_frame()) {
				mink.update_frame(viewer);
			}
		}
		static bool first_clear = true;
		if (first_clear) {
			glClearColor(0.3f, 0.3f, 0.5f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			first_clear = false;
		}
		return false;
	};
	viewer.core.is_animating = false;
	viewer.core.animation_max_fps = 60.;
	viewer.launch(GL_DEPTH_BUFFER_BIT);

	return 0;
}
