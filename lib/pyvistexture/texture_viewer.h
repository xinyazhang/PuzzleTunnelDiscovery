/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef PYVISTEXTURE_TEXTURE_VIEWER_H
#define PYVISTEXTURE_TEXTURE_VIEWER_H

#include <string>
#include <memory>
#include <Eigen/Core>

#include <osr/scene.h>

namespace igl { namespace opengl { namespace glfw {
class Viewer;
} } }

class TextureViewer {
public:
	using Viewer = igl::opengl::glfw::Viewer;

	using EiTex = Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor>;
	using EiTexRGBA = Eigen::Matrix<uint8_t, -1, -1, Eigen::RowMajor>;
	using EiTexFloat = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

	TextureViewer();
	~TextureViewer();
	void loadGeometry(const std::string& obj_fn);

	void initViewer();
	void updateGeometry();
	void updateTexture(const EiTex& r_ch, 
	                   const EiTex& g_ch,
	                   const EiTex& b_ch);
	void updatePointCloud(const Eigen::MatrixXd& pc,
			      const Eigen::MatrixXd& pc_color,
			      float pt_size = -1.0);
	void run();

	// Override this function in Python as callback
	virtual bool key_up(unsigned int key, int modifier);
private:
	osr::Scene scene_;
	Eigen::MatrixXd V_;
	Eigen::MatrixXi F_;
	Eigen::MatrixXd V_uv_;

	std::shared_ptr<Viewer> pviewer_;
};

#endif
