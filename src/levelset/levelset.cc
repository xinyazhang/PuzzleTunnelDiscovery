/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "levelset.h"
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/Util.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <igl/writeOBJ.h>

namespace {
	template <typename Out, typename EigenMatrix>
	std::vector<Out> convert3(const EigenMatrix& m)
	{
		std::vector<Out> ret;
		ret.reserve(m.rows());
		for (int i = 0; i < m.rows(); i++) {
			ret.emplace_back(m(i,0),m(i,1),m(i,2));
		}
		return ret;
	}

	template <typename In, typename EigenMatrix>
	void convert_back3(const std::vector<In>& in, EigenMatrix& m)
	{
		m.resize(in.size(), 3);
		for (size_t i = 0; i < in.size(); i++) {
			m(i, 0) = in[i].x();
			m(i, 1) = in[i].y();
			m(i, 2) = in[i].z();
		}
	}
	template <typename In, typename EigenMatrix>
	void convert_back4(const std::vector<In>& in, EigenMatrix& m)
	{
		m.resize(in.size(), 4);
		for (size_t i = 0; i < in.size(); i++) {
			m(i, 0) = in[i].x();
			m(i, 1) = in[i].y();
			m(i, 2) = in[i].z();
			m(i, 3) = in[i].w();
		}
	}
};

void levelset::generate(
		const Eigen::MatrixXf& inV,
		const Eigen::MatrixXi& inF,
		double mtov_width,
		double vtom_width,
		const std::string& fn
	     )
{
	constexpr double scale_factor = 4.0;
	using std::vector;
	using openvdb::Vec3s;
	using openvdb::Vec3I;
	using openvdb::Vec4I;
	auto scaledVA = inV.array().eval();
	scaledVA *= scale_factor;

	auto V = convert3<Vec3s>(scaledVA);
	auto F = convert3<Vec3I>(inF);
	
	auto tf = openvdb::math::Transform::createLinearTransform();
	auto grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(*tf, V, F, mtov_width);
#if 0
	openvdb::io::File file(fn.c_str());
	file.write({grid});
	file.close();
#endif
	std::vector<Vec3s> OV;
	std::vector<Vec3I> OF;
	std::vector<Vec4I> OQ;

	openvdb::tools::volumeToMesh(*grid, OV, OF, OQ, vtom_width * scale_factor);

	Eigen::MatrixXf eigenV;
	Eigen::MatrixXi eigenF;
	convert_back3(OV, eigenV);
	// convert_back3(OF, eigenF);
	convert_back4(OQ, eigenF);
	{
		auto VA = eigenV.array();
		VA *= 1.0/scale_factor;
		eigenV = VA.matrix();
	}
	igl::writeOBJ(fn, eigenV, eigenF);
}
