#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <Eigen/Core>
#include <unordered_map>
#include <memory>
#include <boost/range/irange.hpp>
#include <vector>

#if 0
class GeometryGroup;

struct IndexedGeometry {
public:
	IndexedGeometry(GeometryGroup* = nullptr);

	int get_local_index();
	int get_global_index();
	void migrate_to(GeometryGroup*);
private:
	int local_index_ = -1;
	int global_index_ = -1;
	GeometryGroup* group_ = nullptr;
};

class GeometryGroup {
public:
	static GeometryGroup* global_instance();
	int get_geometry_index(IndexedGeometry*);

private:
	int gindex_ = 0;
	std::unordered_map<IndexedGeometry*, int> maps_;
};
#endif

typedef Eigen::Vector2d MazeVert;
typedef std::vector<MazeVert, Eigen::aligned_allocator<MazeVert> > MazeVertArray;

class MazeSegment {
public:
	MazeVert v0, v1;
};

class MazeBoundary {
public:
	MazeBoundary(std::istream&);

	MazeSegment& get_prim(int idx);
	MazeVert get_center();
	auto irange() { return boost::irange(0, (int)segs_.size()); }
private:
	std::vector<MazeSegment> segs_;
	MazeVert center_;
};

typedef Eigen::Vector3d ObVertex;

struct ObFace {
	ObFace(ObVertex*, ObVertex*, ObVertex*);
	ObVertex* verts[3] = {nullptr, nullptr, nullptr};
};

class LayerPolygon;

class Obstacle {
public:
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
};

#endif
