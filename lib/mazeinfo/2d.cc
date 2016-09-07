#include "2d.h"

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
