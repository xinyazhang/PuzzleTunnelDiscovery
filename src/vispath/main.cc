#include <iostream>
#include <sstream>
#include <string>
#include <stdint.h>
#include <cstdlib>
#include <unordered_map>
#include <fstream>
#include <chrono>
#include <vector>

#include <Eigen/Core>

#include <osr/unit_world.h>
#include <osr/cdmodel.h>
#include <osr/scene.h>
#include <osr/mesh.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/material_colors.h>

// For ancient libigl
// using Viewer = igl::viewer::Viewer;
// For newer libigl
using Viewer = igl::opengl::glfw::Viewer;

using osr::UnitWorld;
using Clock = std::chrono::high_resolution_clock;

namespace {
UnitWorld uw;
size_t env_data_index;
size_t rob_data_index;
double speed = 1.0;
double tau = 0;

auto last_draw_time = Clock::now();
bool Qs_has_rotation = true;
osr::ArrayOfStates Qs;
Eigen::VectorXd miles;
double total_miles;

bool play = false;
bool first_draw = false;
bool show_all = false;
};

void usage()
{
	std::cerr << R"xxx(Usage: vispath <env .obj file> <robot .obj file> <unitary path file in txt> <speed>)xxx" << std::endl;
}

size_t load_geometry_to_viewer(Viewer& viewer, const uint32_t geo_id, bool overwrite = false)
{
	size_t ret = viewer.selected_data_index;
	if (!overwrite)
		ret = viewer.append_mesh();
	auto cd = uw.getCDModel(geo_id);
	auto& verts = cd->vertices();
	Eigen::Vector3d cmin = verts.colwise().minCoeff();
	Eigen::Vector3d cmax = verts.colwise().maxCoeff();
	viewer.core.camera_center = ((cmin + cmax) * 0.5).cast<float>();
	std::cerr << "Setting camera center to " << viewer.core.camera_center.transpose() << std::endl;
	viewer.data().set_mesh(verts, cd->faces());
	//viewer.data().set_uv(V_uv);
	viewer.data().compute_normals();
	viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;

	return ret;
}

osr::StateVector
interpolate(const osr::ArrayOfStates& qs,
            const Eigen::VectorXd& miles,
            double tau)
{
	int key = -1;
	for (int i = 0; i < qs.rows() - 1; i++) {
		if (miles(i) <= tau && tau < miles(i+1)) {
			key = i;
			break;
		}
	}
	if (key < 0) {
		return qs.row(0);
	}
	double d = miles(key+1) - miles(key);
	double t;
	if (d < 1e-6) 
		t = 0.0;
	else
		t = (tau - miles(key)) / d;
	return osr::interpolate(qs.row(key), qs.row(key+1), t);
}

bool predraw(Viewer& viewer)
{
	auto& rob_data = viewer.data_list[rob_data_index];
	if (show_all) {
		// std::cerr << "Qs rows: " << Qs.rows() << " cols: " << Qs.cols() << std::endl;
		if (Qs_has_rotation) {
			for (int i = 0; i < Qs.rows(); i++) {
				osr::StateVector q = Qs.row(i).transpose();
				rob_data.set_transform(osr::translate_state_to_transform(q).matrix(), i + 1);
			}
		} else {
			int nv = Qs.rows();
			Eigen::MatrixXd pc_color(nv, 3);
			pc_color.col(0).array() = 0.0;
			pc_color.col(1).array() = 1.0;
			pc_color.col(2).array() = 0.0;
			std::cerr << pc_color.row(0) << std::endl;
			auto& env_data = viewer.data_list[env_data_index];
			Eigen::MatrixXd pc = Qs.block(0, 0, nv, 3);
			env_data.set_points(pc, pc_color);
		}
		show_all = false;
	} else {
		if (rob_data.transforms.size() > 1 || rob_data.transforms.count(0) == 0) {
			rob_data.reset_transforms();
		}
	}
	if (!play)
		return false;
	std::cerr << "tau: " << tau << std::endl;
	
	osr::StateVector q = interpolate(Qs, miles, tau);
	osr::StateTrans qt = std::get<0>(osr::decompose(q));
	viewer.core.camera_center = qt.cast<float>() * viewer.core.camera_zoom * viewer.core.camera_base_zoom;
	rob_data.set_transform(osr::translate_state_to_transform(q).matrix());
	Eigen::MatrixXd pc_color(1, 3), pc_point(1, 3);
	pc_point.row(0) = qt;
	pc_color.row(0) << 0.0, 0.0, 1.0;
	// Do NOT set to rob data, because it's translated
	viewer.data_list[env_data_index].set_points(pc_point, pc_color);

	auto now = Clock::now();
	if (first_draw) {
		first_draw = false;
	} else {
		std::chrono::duration<double> sec(now - last_draw_time);
		tau += speed * sec.count();
		if (tau > total_miles) {
			tau = 0.0;
			play = false;
		}
	}
	last_draw_time = now;
	return false;
}

bool key_up(Viewer& viewer, unsigned int key, int modifier)
{
	// std::cerr << key << std::endl;
	if (key == 'p' or key == 'P') {
		play = !play;
		first_draw = true;
	} else if (key == '=') {
		speed *= 2.0;
		std::cout << "current speed: " << speed << std::endl;
	} else if (key == '-') {
		speed /= 2.0;
		std::cout << "current speed: " << speed << std::endl;
	} else if (key == GLFW_KEY_KP_MULTIPLY || ((key == GLFW_KEY_KP_8) && (modifier & GLFW_MOD_SHIFT))) {
		// * enters show_all mode unconditionally Press p to play
		play = false;
		first_draw = true;
		show_all = true;
	}
	return false;
}

void load_path(const std::string& fn)
{
	std::ifstream fin(fn);
	std::vector<osr::StateVector> qs;
	osr::StateVector q;
	std::string line;
	while (!fin.eof()) {
		std::getline(fin, line);
		std::stringstream linein(line);
		int i = 0;
		while (linein >> q(i))
			i++;
		if (i == 0)
			break;
		// i += 1; // 0 indexed index to the dimension
		if (i < 3) {
			for (int j = 0; j < q.rows(); j++)
				std::cerr << q(j) << std::endl;
			throw std::runtime_error("path dimension is: " + std::to_string(i) + ", should be at least 3");
		} else if (i == 3) {
			q(3) = 1.0;
			q(4) = 0.0;
			q(5) = 0.0;
			q(6) = 0.0;
			Qs_has_rotation = false;
		} else if (i < osr::kStateDimension) {
			for (int j = 0; j < q.rows(); j++)
				std::cerr << q(j) << std::endl;
			throw std::runtime_error("path dimension is: " + std::to_string(i) + ", which is incomplete (3 or " + std::to_string(osr::kStateDimension) + ")");
		}
		qs.emplace_back(q);
	}
	Qs.resize(qs.size(), q.rows());
	miles.resize(qs.size(), 1);
	double dist = 0.0;
	for (size_t i = 0; i < qs.size(); i++) {
		if (i > 0) {
			dist += osr::distance(qs[i-1], qs[i]);
		}
		Qs.row(i) = qs[i];
		miles(i) = dist;
	}
	total_miles = dist;
}

int main(int argc, const char* argv[])
{
	if (argc < 5) {
		usage();
		return -1;
	}
	std::string env_fn(argv[1]), rob_fn(argv[2]), txt_fn(argv[3]);
	speed = std::atof(argv[4]);

	load_path(txt_fn);

	uw.loadModelFromFile(env_fn);
	uw.loadRobotFromFile(rob_fn);
	uw.scaleToUnit();
	uw.angleModel(0,0);

	Viewer viewer;
	rob_data_index = load_geometry_to_viewer(viewer, UnitWorld::GEO_ROB, true);
	env_data_index = load_geometry_to_viewer(viewer, UnitWorld::GEO_ENV, false);

	// Default configuration
	viewer.callback_key_up = &key_up;
	viewer.callback_pre_draw = &predraw;
	viewer.selected_data_index = rob_data_index;
	viewer.core.is_animating = true;

	viewer.launch();

	return 0;
}
