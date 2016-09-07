#ifndef MAZEINFO_2D_H
#define MAZEINFO_2D_H

#include <Eigen/Core>
#include <boost/range/irange.hpp>
#include <vector>

typedef Eigen::Vector2d MazeVert;
typedef std::vector<MazeVert, Eigen::aligned_allocator<MazeVert> > MazeVertArray;

class MazeSegment {
public:
	MazeVert v0, v1;
};

class MazeBoundary {
public:
	MazeBoundary(std::istream&);

	MazeSegment& get_prim(int idx); // Get #idx primitive
	MazeVert get_center() const;
	auto irange() { return boost::irange(0, (int)segs_.size()); }

	void get_bbox(MazeVert& minV, MazeVert& maxV) const;
	void merge_bbox(MazeVert& minV, MazeVert& maxV) const;
private:
	std::vector<MazeSegment> segs_;
	MazeVert center_;
};


#endif
