/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "2d.h"
#include <advplyio/ply_write_vfc.h>

using std::endl;

MazeBoundary::MazeBoundary(std::istream& fin)
{
	int size;
	fin >> size;
	segs_.resize(size);
	center_ << 0.0, 0.0;
	for(auto& seg : segs_) {
		fin >> seg.v0.x() >> seg.v0.y();
		fin >> seg.v1.x() >> seg.v1.y();
		center_ += seg.v0;
		center_ += seg.v1;
	}
	center_ /= size*2;
}

MazeSegment& MazeBoundary::get_prim(int idx)
{
	return segs_[idx];
}

MazeVert MazeBoundary::get_center() const
{
	return center_;
}

void MazeBoundary::get_bbox(MazeVert& minV, MazeVert& maxV) const
{
	minV = maxV = segs_.front().v0;
	merge_bbox(minV, maxV);
}

void MazeBoundary::merge_bbox(MazeVert& minV, MazeVert& maxV) const
{
	for (const auto& seg : segs_) {
		for (int i = 0; i < 2; i++) {
			minV(i) = std::min(minV(i), seg.v0(i));
			minV(i) = std::min(minV(i), seg.v1(i));
			maxV(i) = std::max(maxV(i), seg.v0(i));
			maxV(i) = std::max(maxV(i), seg.v1(i));
		}
	}
}

void MazeBoundary::writePLY(std::ostream& fout, Eigen::Vector3d color)
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	convertVF(V, F);
	ply_write_naive_header(fout, V.rows(), F.rows());

	fout.precision(17);
	Eigen::VectorXi C = (color * 255.0).cast<int>();
	for(int i = 0; i < V.rows(); i++) {
		fout << V(i,0) << ' ' << V(i,1) << ' ' << V(i,2) << ' ';
		fout << C(0) << ' ' << C(1) << ' ' << C(2) << endl;
	}
	for(int i = 0; i < F.rows(); i++) {
		fout << F.cols();
		for(int j = 0; j < F.cols(); j++)
			fout << ' ' << F(i, j);
		fout << endl;
	}
}

void MazeBoundary::convertVF(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	MazeVertArray vertlist;
	struct iseg {
		int iv0, iv1;
	};
	std::vector<iseg> iseglist;
	for (const auto& seg : segs_) {
		iseg is = { -1, -1 };
		for (int i = 0; i < int(vertlist.size()); i++) {
			const auto& v = vertlist[i];
			if (v == seg.v0) {
				is.iv0 = i;
			}
			if (v == seg.v1) {
				is.iv1 = i;
			}
		}
		if (is.iv0 < 0) {
			is.iv0 = vertlist.size();
			vertlist.emplace_back(seg.v0);
		}
		if (is.iv1 < 0) {
			is.iv1 = vertlist.size();
			vertlist.emplace_back(seg.v1);
		}
		iseglist.emplace_back(is);
	}
	int nv2d = int(vertlist.size());
	V.resize(vertlist.size() * 2, 3);
	F.resize(iseglist.size() * 2, 3);
	for(int i = 0; i < nv2d; i++) {
		V.row(i) = Eigen::Vector3d(vertlist[i].x(), vertlist[i].y(), 0.0);
		V.row(i + nv2d) = Eigen::Vector3d(vertlist[i].x(), vertlist[i].y(), 1.0);
	}
	for(int i = 0; i < int(iseglist.size()); i++) {
		const auto& is = iseglist[i];
		F.row(2 * i + 0) = Eigen::Vector3i(is.iv0, is.iv1, is.iv0 + nv2d);
		F.row(2 * i + 1) = Eigen::Vector3i(is.iv1, is.iv1 + nv2d, is.iv0 + nv2d);
	}
}
