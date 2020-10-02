/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <unordered_map>
#include <memory>
#include <vector>
#include <mazeinfo/2d.h>

typedef Eigen::Vector3d ObVertex;

struct ObFace {
	ObFace(ObVertex*, ObVertex*, ObVertex*);
	ObVertex* verts[3] = {nullptr, nullptr, nullptr};
};

class LayerPolygon;
class Options;

class Obstacle {
public:
	Obstacle(Options& o);
	void construct(const MazeSegment& wall,
		const MazeSegment& stick,
		const MazeVert& stick_center);
	void build_VF(Eigen::MatrixXd& V, Eigen::MatrixXi& F);
private:
	std::vector<std::unique_ptr<ObVertex> > V_;
	std::vector<ObFace> F_;

	void connect_parallelogram(LayerPolygon&, LayerPolygon&);
	void append_F(const std::vector<ObFace>&);
	void append_V(const std::vector<ObVertex*>&);

	std::unordered_map<ObVertex*, int> vimap_;
	int locate(ObVertex*);
	int vi_ = 0;

	void seal(LayerPolygon&, bool reverse);
	Options& opt;
};

#endif
