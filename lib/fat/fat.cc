#include "fat.h"
#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/util/Util.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetFilter.h>

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

	template <typename Out, typename EigenMatrix>
	std::vector<Out> convert3(const EigenMatrix& m, double scale_factor)
	{
		std::vector<Out> ret;
		ret.reserve(m.rows());
		for (int i = 0; i < m.rows(); i++) {
			ret.emplace_back(m(i,0) * scale_factor,
					m(i,1) * scale_factor,
					m(i,2) * scale_factor);
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
	void convert_back3(const std::vector<In>& in, EigenMatrix& m, double scale_factor)
	{
		m.resize(in.size(), 3);
		for (size_t i = 0; i < in.size(); i++) {
			m(i, 0) = in[i].x() * scale_factor;
			m(i, 1) = in[i].y() * scale_factor;
			m(i, 2) = in[i].z() * scale_factor;
		}
	}

	template <typename In, typename EigenMatrix>
	void concat(const std::vector<In>& in, EigenMatrix& m)
	{
		int base = m.rows();
		m.conservativeResize(m.rows() + in.size(), 3);
		
		for (size_t i = 0; i < in.size(); i++) {
			m(base + i, 0) = in[i].x();
			m(base + i, 1) = in[i].y();
			m(base + i, 2) = in[i].z();
		}
	}

	template <typename In, typename EigenMatrix>
	void convert_quads(const std::vector<In>& in, EigenMatrix& m)
	{
		m.resize(2 * in.size(), 3);
		for (size_t i = 0; i < in.size(); i++) {
			m(2 * i, 0) = in[i].x();
			m(2 * i, 2) = in[i].y();
			m(2 * i, 1) = in[i].z();
			m(2 * i + 1, 0) = in[i].x();
			m(2 * i + 1, 2) = in[i].z();
			m(2 * i + 1, 1) = in[i].w();
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

void fat::initialize()
{
	openvdb::initialize();
}

void fat::mkfatter(
			const Eigen::MatrixXf& inV,
			const Eigen::MatrixXi& inF,
			double width,
			Eigen::MatrixXf& outV,
			Eigen::MatrixXi& outF,
			bool trianglize,
			double scale_factor
		)
{
	using std::vector;
	using openvdb::Vec3s;
	using openvdb::Vec3I;
	using openvdb::Vec4I;
	using openvdb::TreeAdapter;
	using openvdb::math::Transform;
	using openvdb::tools::meshToLevelSet;
	using openvdb::tools::volumeToMesh;
	using openvdb::tools::erodeVoxels;

	auto V = convert3<Vec3s>(inV);
	auto F = convert3<Vec3I>(inF);
	
	auto tf = Transform::createLinearTransform(1.0/scale_factor);
	double halfwidth = std::abs(width) * scale_factor;
#if 0
	if (halfwidth <= 0.0)
		halfwidth = 2.0;
#endif
	auto grid = meshToLevelSet<openvdb::FloatGrid>(*tf, V, F, halfwidth);
	std::vector<Vec3s> OV;
	std::vector<Vec3I> OF;
	std::vector<Vec4I> OQ;

	openvdb::tools::LevelSetFilter<openvdb::FloatGrid> lsf(*grid);
	lsf.offset(-width);
	volumeToMesh(*grid, OV, OF, OQ, 0.0, 1.0);

	// convert_back3(OV, outV, 1.0/scale_factor);
	convert_back3(OV, outV);
	if (trianglize)
		convert_quads(OQ, outF);
	else
		convert_back4(OQ, outF);
	concat(OF, outF);
}
