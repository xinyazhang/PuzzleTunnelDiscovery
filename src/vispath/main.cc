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
bool load_all = false;

bool anchor_mode = false;
struct Anchor {
	size_t data_index;
	size_t geometry_id;
	Eigen::Vector3d bary;
	Eigen::Vector3d unit_surface_point;
	Eigen::Vector3d unit_normal;

	Eigen::MatrixXd colors;

	double angle = 0.0;
};

std::unordered_map<size_t, Anchor> anchors;
bool trajector_drawn = false;
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

bool update_trajectory_visualization(Viewer& viewer)
{
	if (anchors.find(rob_data_index) == anchors.end())
		return false;
	Eigen::Vector4d hanchor;
	hanchor(0) = anchors[rob_data_index].unit_surface_point(0);
	hanchor(1) = anchors[rob_data_index].unit_surface_point(1);
	hanchor(2) = anchors[rob_data_index].unit_surface_point(2);
	hanchor(3) = 1.0;
	std::vector<Eigen::Vector4d> std_pts;
	Eigen::MatrixXd e3_pts, pc_color;
	Eigen::MatrixXi e3_edges;
	double t = 0; // do NOT use tau, which is global
	while (t < total_miles) {
		osr::StateVector st = osr::path_interpolate(Qs, miles, t);
		std_pts.emplace_back(osr::translate_state_to_transform(st) * hanchor);
		std::cerr << "Points: " << std_pts.size() << std::endl;
		t += speed;
	}
	e3_pts.resize(std_pts.size(), 3);
	e3_edges.resize(e3_pts.rows() - 1, 2);
	pc_color.resize(1, 3);
	pc_color.row(0) << 0.0, 1.0, 0.0;
	for (size_t i = 0; i < std_pts.size(); i++) {
		e3_pts.row(i) = std_pts[i].head<3>();
	}
	for (size_t i = 0; i < e3_pts.rows() - 1; i++) {
		std::cerr << "edges: " << i << std::endl;
		e3_edges.row(i) << i, i + 1;
	}
	std::cerr << "setting edges" << std::endl;
	// viewer.data_list[env_data_index].set_points(e3_pts, pc_color);
	viewer.data_list[env_data_index].set_edges(e3_pts, e3_edges, pc_color);
	trajector_drawn = true;
	return true;
}

bool predraw(Viewer& viewer)
{
	auto& rob_data = viewer.data_list[rob_data_index];
	// std::cerr << "rob_data.transforms.size: " << rob_data.transforms.size() << std::endl;
	if (load_all) {
		std::cerr << "Qs rows: " << Qs.rows() << " cols: " << Qs.cols() << " has rotation: " << Qs_has_rotation << std::endl;
		if (Qs_has_rotation) {
			for (int i = 0; i < Qs.rows(); i++) {
				osr::StateVector q = Qs.row(i).transpose();
				rob_data.set_transform(osr::translate_state_to_transform(q).matrix(), i + 1);
			}
			std::cerr << "rob_data.transforms.size: " << rob_data.transforms.size() << std::endl;
			viewer.data().dirty = igl::opengl::MeshGL::DIRTY_ALL;
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
		load_all = false;
	}
	if (!show_all) {
		if (rob_data.transforms.size() > 1 || rob_data.transforms.count(0) == 0) {
			rob_data.reset_transforms();
		}
	}
	if (!play)
		return false;

	osr::StateVector q = osr::path_interpolate(Qs, miles, tau);
	osr::StateTrans qt = std::get<0>(osr::decompose(q));
	viewer.core.camera_center = qt.cast<float>() * viewer.core.camera_zoom * viewer.core.camera_base_zoom;
	rob_data.set_transform(osr::translate_state_to_transform(q).matrix());
	if (!trajector_drawn) {
		Eigen::MatrixXd pc_color(1 + anchors.size(), 3), pc_point(1 + anchors.size(), 3);
		pc_point.row(0) = qt;
		pc_color.row(0) << 0.0, 0.0, 1.0;
		size_t i = 1;
		for (const auto& anchor_pair : anchors) {
			pc_point.row(i) = anchor_pair.second.unit_surface_point;
			pc_color.row(i) << 0.0, 1.0, 0.0;
			// std::cerr << "visualize anchor point " << pc_point.row(i) << std::endl;
			i++;
		}
		// Do NOT set to rob data, because it's translated
		viewer.data_list[env_data_index].set_points(pc_point, pc_color);
	}
	bool valid = uw.isValid(q);
	std::cerr << "tau: " << tau << "\tvalid: " << valid << std::endl;
#if 0
	if (valid) {
		using namespace igl;
		rob_data.uniform_colors(Eigen::Vector3d(GOLD_AMBIENT[0], GOLD_AMBIENT[1], GOLD_AMBIENT[2]),
		                        Eigen::Vector3d(GOLD_DIFFUSE[0], GOLD_DIFFUSE[1], GOLD_DIFFUSE[2]),
		                        Eigen::Vector3d(GOLD_SPECULAR[0], GOLD_SPECULAR[1], GOLD_SPECULAR[2]));
	} else {
		using namespace igl;
		rob_data.uniform_colors(Eigen::Vector3d(GOLD_AMBIENT[0], GOLD_AMBIENT[1], GOLD_AMBIENT[2]),
		                        Eigen::Vector3d(FAST_RED_DIFFUSE[0], FAST_RED_DIFFUSE[1], FAST_RED_DIFFUSE[2]),
		                        Eigen::Vector3d(GOLD_SPECULAR[0], GOLD_SPECULAR[1], GOLD_SPECULAR[2]));
	}
#endif

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
		if (modifier & GLFW_MOD_CONTROL) {
			if (update_trajectory_visualization(viewer))
				return false;
		}
		play = !play;
		first_draw = true;
	} else if (key == '=') {
		speed *= 2.0;
		std::cout << "current speed: " << speed << std::endl;
	} else if (key == '-') {
		speed /= 2.0;
		std::cout << "current speed: " << speed << std::endl;
	} else if (key == GLFW_KEY_KP_MULTIPLY || ((key == GLFW_KEY_8) && (modifier & GLFW_MOD_SHIFT))) {
		// * enters show_all mode unconditionally Press p to play
		play = false;
		first_draw = true;
		show_all = !show_all;
		load_all = show_all;
	} else if (key == 'k' or key == 'K') {
		// "K": Switch between 'anchor selection' and 'anchor align' mode.
		anchor_mode = !anchor_mode;
		if (anchor_mode) {
			auto& rob_data = viewer.data_list[rob_data_index];
			osr::StateVector q;
			osr::state_vector_set_identity(q);
			rob_data.set_transform(osr::translate_state_to_transform(q).matrix());
		}
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
		while (i < osr::kStateDimension && linein >> q(i))
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

bool mouse_up(Viewer& viewer, int button, int modifier)
{
	// std::cerr << "modifier " << modifier << std::endl;
	// Anchor mode only set
	if (!anchor_mode) {
		return false;
	}
	if (!(modifier & GLFW_MOD_CONTROL))
		return false;
	// Update the anchor point
	int fid;
	Eigen::Vector3d bc;
	double x = viewer.current_mouse_x;
	double y = viewer.core.viewport(3) - viewer.current_mouse_y;
	auto& rob_data = viewer.data_list[rob_data_index];
	const auto& V = rob_data.V;
        const auto& F = rob_data.F;
        bool unproj = igl::unproject_onto_mesh(Eigen::Vector2f(x,y),
	                             viewer.core.view * viewer.data().transforms[0],
	                             viewer.core.proj,
	                             viewer.core.viewport,
	                             V, F,
	                             fid, bc);

	std::cerr << "unproject result: " << unproj << " from (" << x << ", " << y << ")" << std::endl;

	if (unproj) {
		Anchor& ac = anchors[rob_data_index];
		auto cd = uw.getCDModel(UnitWorld::GEO_ROB);
		ac.unit_normal = cd->faceNormals().row(fid);
		ac.bary = bc;

		ac.unit_surface_point =
			  bc(0) * V.row(F(fid, 0))
			+ bc(1) * V.row(F(fid, 1))
			+ bc(2) * V.row(F(fid, 2));
		rob_data.dirty |= igl::opengl::MeshGL::DIRTY_ALL;

		Eigen::MatrixXd pc_color(anchors.size(), 3), pc_point(anchors.size(), 3);
		size_t i = 0;
		for (const auto& anchor_pair : anchors) {
			pc_point.row(i) = anchor_pair.second.unit_surface_point;
			pc_color.row(i) << 0.0, 1.0, 0.0;
			i++;
		}
		// Do NOT set to rob data, because it's translated
		viewer.data_list[env_data_index].set_points(pc_point, pc_color);
	}
	return false;
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
	viewer.callback_mouse_up = &mouse_up;
	viewer.selected_data_index = rob_data_index;
	viewer.core.is_animating = true;

	viewer.launch();

	return 0;
}
