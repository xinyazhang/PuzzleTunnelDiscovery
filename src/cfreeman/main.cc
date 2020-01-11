#include <iostream>
#include <string>
#include <tuple>
#include <stdint.h>
#include <cstdlib>
#include <unordered_map>
#include <fstream>

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

namespace {
UnitWorld uw;
size_t env_data_index;
size_t rob_data_index;
bool anchor_mode = false;
bool ctrl_pressed = false;
double magnitude = 1e-1; // magnitude of margins, translations, rotations etc.
bool manual_mode = false;
const char* AXIS_NAME[] = {"X", "Y", "Z"};
int manual_axis = 0;
bool manual_sforce = false;
bool manual_guard = false;

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

std::ofstream fout;
osr::StateVector latest_state;
osr::StateVector identity;
osr::StateTrans identity_trans;
osr::StateQuat identity_rot;
bool is_latest_state_free = false;
};

void usage()
{
	std::cerr << R"xxx(Usage: c-freeman <env .obj file> <robot .obj file> [7 numbers for unitary state]
 K: Switch between 'anchor selection' and 'anchor align' mode.
 M: Switch to 'Manual' mode
 Ctrl+S: Save current c-free state to cfreeman.out
 F: Show lines
 T: Show textures
 -/=: increase/decrease the margin by 2x
 Ctrl+mouse wheel: rotate the puzzle
 PgUp/PgDn: change the rotation/translation axis
)xxx" << std::endl;
}

size_t load_geometry_to_viewer(Viewer& viewer, const uint32_t geo_id, bool overwrite = false)
{
#if 0
	auto scene = uw.getScene(geo_id);
	if (scene->getNumberOfMeshes() > 1) {
		std::cerr << "vistexture: do not support multi-mesh models for now\n";
		return -1;
	}
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	Eigen::MatrixXd V_uv;
	std::function<void(std::shared_ptr<const osr::Mesh> mesh)> visitor = [&V, &F, &V_uv](std::shared_ptr<const osr::Mesh> mesh) {
		const auto& mV = mesh->getVertices();
		const auto& mF = mesh->getIndices();
		V.resize(mV.size(), 3);
		for (size_t i = 0; i < mV.size(); i++) {
			const auto& v = mV[i].position;
			V.row(i) << v[0], v[1], v[2];
		}

		F.resize(mF.size() / 3, 3);
		for (size_t i = 0; i < mF.size() / 3; i++) {
			F.row(i) << mF[3 * i + 0], mF[3 * i + 1], mF[3 * i + 2];
		}
		Eigen::MatrixXd uv = mesh->getUV().cast<double>();
		V_uv.resize(uv.rows(), 2);
		V_uv.col(1) = uv.col(0);
		V_uv.col(0) = uv.col(1);
	};
	scene->visitMesh(visitor);
#endif

	size_t ret = viewer.selected_data_index;
	if (!overwrite)
		ret = viewer.append_mesh();
	auto cd = uw.getCDModel(geo_id);
	viewer.data().set_mesh(cd->vertices(), cd->faces());
	//viewer.data().set_uv(V_uv);
	viewer.data().compute_normals();

	anchors[ret].data_index = ret;
	anchors[ret].geometry_id = geo_id;
	anchors[ret].colors = Eigen::MatrixXd::Constant(viewer.data().F.rows(),3,1);

	viewer.data().set_colors(anchors[ret].colors);

	return ret;
}

void update_anchors_visualization(Viewer& viewer)
{
#if 0
	if (!anchor_mode) {
		// clear the points
		viewer.data_list[rob_data_index].points.resize(0, 0);
		return ;
	}
#endif
#if 1
	Eigen::MatrixXd pc_color, pc_point;
	pc_color.setZero(2, 3);
	pc_point.setZero(2, 3);
	pc_point.row(0) = anchors[env_data_index].unit_surface_point;
	pc_color.row(0) << 1.0, 0.0, 0.0;
	pc_point.row(1) = anchors[rob_data_index].unit_surface_point;
	pc_color.row(1) << 0.0, 0.0, 1.0;

	viewer.data_list[rob_data_index].set_points(pc_point, pc_color);
#endif
}

void update_latest_state(Viewer& viewer, const osr::StateVector& q)
{
	bool valid = uw.isValid(q);

	// Visualization
	// 1. Env model is already loaded, skipped
	// 2. Update the Rob model with generated q
	auto& rob_data = viewer.data_list[rob_data_index];
#if 1
	rob_data.set_transform(osr::translate_state_to_transform(q).matrix());
#else
	UnitWorld::VMatrix V;
	UnitWorld::FMatrix F;

	std::tie(V, F) = uw.getRobotGeometry(q, true);
	rob_data.V = V;
	rob_data.F = F;

	rob_data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
#endif
	if (valid) {
		rob_data.set_colors(anchors[rob_data_index].colors);
	} else {
		using namespace igl;
		rob_data.uniform_colors(Eigen::Vector3d(GOLD_AMBIENT[0], GOLD_AMBIENT[1], GOLD_AMBIENT[2]),
		                        Eigen::Vector3d(FAST_RED_DIFFUSE[0], FAST_RED_DIFFUSE[1], FAST_RED_DIFFUSE[2]),
		                        Eigen::Vector3d(GOLD_SPECULAR[0], GOLD_SPECULAR[1], GOLD_SPECULAR[2]));
		auto tup = uw.intersectingSegments(q);
		Eigen::Matrix<int, -1, 2> findices = std::get<3>(tup);
		for (int i = 0; i < findices.rows(); i++) {
			int f = findices(i, 1);
			rob_data.F_material_ambient.row(f) = Eigen::Vector3d(CYAN_AMBIENT[0], CYAN_AMBIENT[1], CYAN_AMBIENT[2]);
			rob_data.F_material_diffuse.row(f) = Eigen::Vector3d(FAST_BLUE_DIFFUSE[0], FAST_BLUE_DIFFUSE[1], FAST_BLUE_DIFFUSE[2]);
			rob_data.F_material_specular.row(f) = Eigen::Vector3d(CYAN_SPECULAR[0], CYAN_SPECULAR[1], CYAN_SPECULAR[2]);
		}
	}
	latest_state = q;
	is_latest_state_free = valid;
	// is_latest_state_free = true; // debugging...
}

void update_cfree_visualization(Viewer& viewer, double margin)
{
	margin *= 1e-3; // Correct margin's scaling
	using namespace osr;

	StateTrans rob_surface_point = anchors[rob_data_index].unit_surface_point;
	StateTrans rob_surface_normal = anchors[rob_data_index].unit_normal;
	StateTrans env_surface_point = anchors[env_data_index].unit_surface_point;
	StateTrans env_surface_normal = anchors[env_data_index].unit_normal;
	double theta = anchors[rob_data_index].angle;

	StateTrans rob_o = rob_surface_point + rob_surface_normal * margin;
        StateTrans env_o = env_surface_point + env_surface_normal * margin;
        StateVector q; // return value

        // Configuration (Q) sampling algorithm:
        // 1. Rotate Robot so that rob_surface_normal matches **negatived** env_surface_normal
        // 2. Rotate Robot with random angle axis (omega, env_surface_normal)
        // 3. Translate the rotated rob_surface_point to env_surface_point
        using Quat = Eigen::Quaternion<StateScalar>;
        using AA = Eigen::AngleAxis<StateScalar>;
        // Step 1 Rotation
        Quat rot_1;
        rot_1.setFromTwoVectors(rob_surface_normal, -env_surface_normal);
	// Step 2 Rotation
	Quat rot_2(AA(theta, env_surface_normal));
	Quat rot_accum = rot_2 * rot_1;

	// Step 3 Translation
	StateTrans trans = env_o - (rot_accum * rob_o);
	q = compose(trans, rot_accum);
	update_latest_state(viewer, q);
}

bool mouse_up(Viewer& viewer, int button, int modifier)
{
	// std::cerr << "modifier " << modifier << std::endl;
	// Anchor mode only set
#if 1
	if (anchor_mode && viewer.selected_data_index == env_data_index) {
		return false;
	}
	if (!anchor_mode && viewer.selected_data_index == rob_data_index) {
		return false;
	}
#endif
	if (!(modifier & GLFW_MOD_CONTROL))
		return false;
	// Update the anchor point
	int fid;
	Eigen::Vector3d bc;
	double x = viewer.current_mouse_x;
	double y = viewer.core().viewport(3) - viewer.current_mouse_y;
	const auto& V = viewer.data().V;
        const auto& F = viewer.data().F;
        bool unproj = igl::unproject_onto_mesh(Eigen::Vector2f(x,y),
	                             viewer.core().view * viewer.data().transforms[0],
	                             viewer.core().proj,
	                             viewer.core().viewport,
	                             V, F,
	                             fid, bc);

	std::cerr << "unproject result: " << unproj << " from (" << x << ", " << y << ")" << std::endl;

	if (unproj) {
#if 0
		Anchor& ac = anchors[viewer.selected_data_index];
		ac.bary = bc;

		Eigen::Vector3d tri_v[3];
		tri_v[0] = V.row(F(fid, 0));
		tri_v[1] = V.row(F(fid, 1));
		tri_v[2] = V.row(F(fid, 2));
#if 1
		ac.unit_surface_point = bc(0) * tri_v[0];
		ac.unit_surface_point += bc(1) * tri_v[1];
		ac.unit_surface_point += bc(2) * tri_v[2];
#else
		ac.unit_surface_point = tri_v[0];
#endif
#else
		Anchor& ac = anchors[viewer.selected_data_index];
		auto cd = uw.getCDModel(ac.geometry_id);
		ac.unit_normal = cd->faceNormals().row(fid);
		ac.bary = bc;
#if 0
		std::cerr << "bc: " << bc.transpose() << std::endl;
		std::cerr << "faces: "
			<< F(fid, 0) << " "
			<< F(fid, 1) << " "
			<< F(fid, 2) << " " << std::endl;
		std::cerr << "verts: " << "\n"
			<< V.row(F(fid, 0)) << "\n"
			<< V.row(F(fid, 1)) << "\n"
			<< V.row(F(fid, 2)) << "\n" << std::endl;
#endif

#if 0
		Eigen::Vector3d tri_v[3];
		tri_v[0] = V.row(F(fid, 0));
		tri_v[1] = V.row(F(fid, 1));
		tri_v[2] = V.row(F(fid, 2));
		ac.unit_surface_point = bc(0) * tri_v[0];
		ac.unit_surface_point += bc(1) * tri_v[1];
		ac.unit_surface_point += bc(2) * tri_v[2];
#endif
		ac.unit_surface_point =
			  bc(0) * V.row(F(fid, 0))
			+ bc(1) * V.row(F(fid, 1))
			+ bc(2) * V.row(F(fid, 2));
		// std::cerr << "surface: " << ac.unit_surface_point << std::endl;

#if 0
		auto &C = anchors[viewer.selected_data_index].colors;
		C.row(fid) << 1,0,0;
		viewer.data().set_colors(C);
#endif
#endif
		update_anchors_visualization(viewer);
	}

	if (!anchor_mode)
		update_cfree_visualization(viewer, magnitude);

	return false;
}

bool mouse_scroll(Viewer& viewer, float delta_y)
{
	// std::cerr << "scroll " << delta_y << std::endl;
	// std::cerr << "ctrl_pressed " << ctrl_pressed << std::endl;
	if (manual_mode) {
		osr::AngleAxisVector aa;
		osr::StateTrans tr;
		tr.setZero();
		aa.setZero();
		if (ctrl_pressed)
			aa(manual_axis) = 1.0 * delta_y * magnitude;
		else
			tr(manual_axis) = 1.0 * delta_y * magnitude;
		if (!manual_guard) {
			osr::StateVector q = osr::apply(latest_state, tr, aa);
			update_latest_state(viewer, q);
		} else {
			auto tup = uw.transitStateBy(latest_state, tr, aa, magnitude / 16.0);
			update_latest_state(viewer, std::get<0>(tup));
		}
		return true;
		// is_latest_state_free = uw.isValid(latest_state);
		// auto& rob_data = viewer.data_list[rob_data_index];
		// rob_data.set_transform(osr::translate_state_to_transform(latest_state).matrix());
	} else {
		if (!ctrl_pressed)
			return false;
		anchors[rob_data_index].angle += delta_y / 180.0 * M_PI;
		std::cerr << "scroll to " << anchors[rob_data_index].angle << std::endl;

		update_cfree_visualization(viewer, magnitude);
		return true;
	}
}

bool key_down(Viewer& viewer, unsigned int key, int modifier)
{
#if 0
	if (modifier & GLFW_MOD_CONTROL)
		ctrl_pressed = true;
	std::cerr << "key " << key << std::endl;
	std::cerr << "modifier " << modifier << std::endl;
#else
	if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)
		ctrl_pressed = true;
	// std::cerr << "ctrl_pressed " << ctrl_pressed << std::endl;
#endif
	return false;
}

bool key_up(Viewer& viewer, unsigned int key, int modifier)
{
#if 0
	if (key == 'f' or key == 'F') {
		viewer.core.show_lines = not viewer.core.show_lines;
	} else if (key == 't' or key == 'T') {
		viewer.core.show_lines = not viewer.core.show_texture;
	}
#else
	if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL)
		ctrl_pressed = false;
	// std::cerr << "ctrl_pressed " << ctrl_pressed << std::endl;
	if (key < 128)
		std::cerr << (char)key << " pressed " << std::endl;
	else
		std::cerr << key << " pressed " << std::endl;
	if (key == 'k' or key == 'K') {
		// "K": Switch between 'anchor selection' and 'anchor align' mode.
		anchor_mode = !anchor_mode;
		if (anchor_mode) {
			// Reset robot's V and F
			auto& rob_data = viewer.data_list[rob_data_index];
			auto cd = uw.getCDModel(UnitWorld::GEO_ROB);
			rob_data.V = cd->vertices();
			rob_data.F = cd->faces();
			rob_data.dirty = igl::opengl::MeshGL::DIRTY_ALL;

			viewer.selected_data_index = rob_data_index;
		} else {
			viewer.selected_data_index = env_data_index;
			update_cfree_visualization(viewer, magnitude);
		}
	} else if ((modifier & GLFW_MOD_CONTROL) and (key == 's' or key == 'S')) {
		if (!is_latest_state_free) {
			std::cerr << "Not a valid state" << std::endl;
			return false;
		}
		std::cerr << latest_state.transpose() << std::endl;
		fout << latest_state.transpose() << std::endl;
		fout.flush();
	} else if (key == 'f' or key == 'F') {
		if (ctrl_pressed)
			manual_sforce = true;
		else
			viewer.data().show_lines = not viewer.data().show_lines;
	} else if (key == 't' or key == 'T') {
		viewer.data().show_texture = not viewer.data().show_texture;
	} else if (key == '-') {
		magnitude *= 0.5;
		if (!manual_mode)
			update_cfree_visualization(viewer, magnitude);
	} else if (key == '=') {
		magnitude *= 2.0;
		if (!manual_mode)
			update_cfree_visualization(viewer, magnitude);
	} else if (key == 'M' or key == 'm') {
		manual_mode = !manual_mode;
		if (manual_mode) {
			std::cout << "Manual mode enabled\n";
			std::cout << "Current axis: " << AXIS_NAME[manual_axis] << std::endl;
		} else {
			std::cout << "Manual mode disabled\n";
		}
	} else if (key == 'G' or key == 'g') {
		manual_guard = !manual_guard;
		std::cerr << "Manual guard: " << (manual_guard ? "Enabled" : "Disabled")
		          << std::endl;
	} else if (key == GLFW_KEY_PAGE_UP) {
		manual_axis += 1;
		manual_axis %= 3;
	} else if (key == GLFW_KEY_PAGE_DOWN) {
		manual_axis -= 1;
		manual_axis += 3;
		manual_axis %= 3;
	} else {
		return false;
	}
#endif
	return true;
}

bool predraw(Viewer& viewer)
{
	if (manual_mode && manual_sforce && !is_latest_state_free) {
		constexpr double STIFFNESS = 1e6;
		constexpr double D_TIME = 1e-3;
		using ArrayOfPoints = osr::ArrayOfPoints;
		ArrayOfPoints sbegins, sends;
		Eigen::Matrix<osr::StateScalar, -1, 1> smags, fmags;
		Eigen::Matrix<int, -1, 2> findices;
		std::tie(sbegins, sends, smags, findices) = uw.intersectingSegments(latest_state);
		fmags = STIFFNESS * smags;
		ArrayOfPoints fposs, fdirs;
		std::tie(fposs, fdirs) = uw.forceDirectionFromIntersectingSegments(sbegins, sends, findices);
		osr::StateVector q = uw.pushRobot(latest_state,
		                                  fposs, fdirs, fmags, 1.0, D_TIME,
		                                  false);
		update_latest_state(viewer, q);
		if (is_latest_state_free)
			manual_sforce = false;
	}
	// Set visibility according to anchor mode
	// Anchor mode: only robot is shown/selected
	// Non-anchor mode: all models are shown
	for (auto& data : viewer.data_list) {
		data.show_faces = !anchor_mode;
		// data.show_lines = !anchor_mode;
		data.show_overlay = !anchor_mode;
	}
	auto& rob_data = viewer.data_list[rob_data_index];
	rob_data.show_faces = true;
	// rob_data.show_lines = true;
	rob_data.show_overlay = true;

	if (anchor_mode)
		viewer.selected_data_index = rob_data_index;
	return false;
}


int main(int argc, const char* argv[])
{
	if (argc < 3) {
		usage();
		return -1;
	}
	std::string env_fn(argv[1]), rob_fn(argv[2]);

	uw.loadModelFromFile(env_fn);
	uw.loadRobotFromFile(rob_fn);
	uw.scaleToUnit();
	uw.angleModel(0,0);
	osr::state_vector_set_identity(latest_state);
	osr::state_vector_set_identity(identity);
	std::tie(identity_trans, identity_rot) = osr::decompose(identity);
	if (argc == 1 /* cmd */ + 2 /* models */ + 7 /* states */) {
		for (int i = 3; i < 3 + 7; i++) {
			latest_state(i - 3) = std::atof(argv[i]);
		}
	}

	Viewer viewer;
	env_data_index = load_geometry_to_viewer(viewer, UnitWorld::GEO_ENV, true);
	rob_data_index = load_geometry_to_viewer(viewer, UnitWorld::GEO_ROB);

	// Default configuration
	viewer.selected_data_index = rob_data_index;

	// We need track the ctrl status
	viewer.callback_key_up = &key_up;
	viewer.callback_key_down = &key_down;
	viewer.callback_pre_draw = &predraw;
	viewer.callback_mouse_up = &mouse_up;
	viewer.callback_mouse_scroll = &mouse_scroll;

	fout.open("cfreeman.out", std::ios_base::app);
	fout << std::endl;
	fout.precision(17);
	viewer.launch();
	fout.close();

	return 0;
}
