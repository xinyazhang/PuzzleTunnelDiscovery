#include <iostream>
#include <string>
#include <stdint.h>

#include <Eigen/Core>

#include <osr/scene.h>
#include <osr/mesh.h>
#include <osr/pngimage.h>

#include <igl/viewer/Viewer.h>
#include <igl/writeOBJ.h>

using Viewer = igl::viewer::Viewer;
// For newer libigl
// using Viewer = igl::opengl::glfw::Viewer;

void usage()
{
	std::cerr << R"xxx(Usage: vistexture <.obj file> <.png texture>)xxx" << std::endl;

}

bool key_up(Viewer& viewer, unsigned char key, int modifier)
{
#if 1
	if (key == 'f' or key == 'F') {
		viewer.core.show_lines = not viewer.core.show_lines;
	} else if (key == 't' or key == 'T') {
		viewer.core.show_lines = not viewer.core.show_texture;
	}
#else
	if (key == 'f' or key == 'F') {
		viewer.data().show_lines = not viewer.data().show_lines;
	} else if (key == 't' or key == 'T') {
		viewer.data().show_lines = not viewer.data().show_texture;
	}
#endif
	return false;
}

int main(int argc, const char* argv[])
{
	if (argc < 3) {
		usage();
		return -1;
	}
	std::string obj_fn(argv[1]), tex_fn(argv[2]);
	osr::Scene scene;
	scene.load(obj_fn);
	if (scene.getNumberOfMeshes() > 1) {
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
	scene.visitMesh(visitor);

	int tex_w, tex_h, pc;
	auto tex_data = osr::readPNG(tex_fn.c_str(), tex_w, tex_h, &pc);

	using EiTex = Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor>;
	using EiTexRGBA = Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor>;
	using EiTexFloat = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
	EiTex r_ch = EiTex::Constant(tex_w, tex_h, 32);
	EiTex b_ch = EiTex::Constant(tex_w, tex_h, 32);

	EiTexRGBA rgba = EiTexRGBA::Map(tex_data.data(), tex_w * tex_h, pc);
	EiTex g_ch_vector = rgba.col(1);
	Eigen::Map<EiTex> g_ch_map(g_ch_vector.data(), tex_w, tex_h);
	EiTex g_ch = g_ch_map;

	Viewer viewer;
#if 1
	viewer.data.set_mesh(V, F);
	viewer.data.set_uv(V_uv);
	viewer.data.compute_normals();
	viewer.data.set_texture(r_ch, g_ch, b_ch);
	// Default configuration
	viewer.core.show_texture = true;
#else // Newer libigl
	viewer.data().set_mesh(V, F);
	viewer.data().set_uv(V_uv);
	viewer.data().set_texture(r_ch, g_ch, b_ch);
	viewer.data().compute_normals();
	// Default configuration
	viewer.data().show_lines = false;
	viewer.data().show_texture = true;
#endif

	viewer.callback_key_up = &key_up;

	viewer.launch();

	return 0;
}
