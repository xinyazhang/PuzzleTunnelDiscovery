#include "texture_viewer.h"
#include <igl/opengl/glfw/Viewer.h>
#include <osr/scene.h>
#include <osr/mesh.h>


TextureViewer::TextureViewer()
{
}

TextureViewer::~TextureViewer()
{
}

void TextureViewer::loadGeometry(const std::string& obj_fn)
{
	scene_.load(obj_fn);
	Eigen::MatrixXd& V = V_;
	Eigen::MatrixXi& F = F_;
	Eigen::MatrixXd& V_uv = V_uv_;
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
	scene_.visitMesh(visitor);
	updateGeometry();
}

void TextureViewer::initViewer()
{
	if (pviewer_)
		return ;
	pviewer_.reset(new Viewer);
	updateGeometry();
	auto key_up_func = [=](Viewer& viewer, unsigned int key, int modifier) {
		return this->key_up(key, modifier);
	};
	pviewer_->callback_key_up = key_up_func;
}


void TextureViewer::updateGeometry()
{
	if (!pviewer_)
		return ;
	Viewer& viewer = *pviewer_;
	viewer.data().set_mesh(V_, F_);
	viewer.data().set_uv(V_uv_);
	viewer.data().compute_normals();
}

void TextureViewer::updateTexture(const EiTex& r_ch, 
                                  const EiTex& g_ch,
                                  const EiTex& b_ch)
{
	if (!pviewer_)
		return ;
	Viewer& viewer = *pviewer_;
	viewer.data().set_texture(r_ch, g_ch, b_ch);
	viewer.data().show_texture = true;
}

void TextureViewer::updatePointCloud(const Eigen::MatrixXd& pc,
                                     const Eigen::MatrixXd& pc_color,
                                     float pt_size)
{
	if (!pviewer_)
		return ;
	Viewer& viewer = *pviewer_;
	viewer.data().set_points(pc, pc_color);
	if (pt_size > 0) {
		viewer.data().point_size = pt_size;
	}
}

void TextureViewer::run()
{
	if (!pviewer_)
		return ;
	Viewer& viewer = *pviewer_;
	viewer.launch();
}

bool TextureViewer::key_up(unsigned int, int)
{
	return true;
}
