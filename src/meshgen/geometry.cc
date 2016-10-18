#include "geometry.h"
#include <Eigen/Geometry> 
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "options.h"

using std::endl;

constexpr int kNInterp = 16; //16;
constexpr int kNSlices = 32; //32;
constexpr int kNLayers = kNSlices + 1;
constexpr double kDeltaTheta = M_PI/32; //2 * M_PI/kNSlices;
constexpr double kBaseTheta = 0; // 0;
constexpr double kEndTheta = 2 * M_PI; // 2 * M_PI; //2 * M_PI/kNSlices;
constexpr double kEpsilon = 1e-6;
constexpr double kWallEpsilon= 1.0/1024.0; //0.01;
//constexpr double kExpandRatio = 1.0+1.0/1024.0; // 1 + small number
constexpr double kExpandRatio = 1.0;

ObFace::ObFace(ObVertex* v0, ObVertex* v1, ObVertex* v2)
{
	verts[0] = v0;
	verts[1] = v1;
	verts[2] = v2;
}

struct LayerPolygon {
	std::vector<ObVertex*>& get_boundary(); // Interpolation happens here.
	std::vector<ObVertex*> get_negative_strip();
	std::vector<ObVertex*> get_positive_strip();
	std::vector<ObVertex*> get_left_side();
	std::vector<ObVertex*> get_right_side();

	double rotate_angle() const { return theta_; } // Angle relative to original position, same as values stored in Z axis
	LayerPolygon(MazeVertArray vs, double zcoord, const MazeSegment&, double); // Singular
	LayerPolygon(MazeVertArray vs, double zcoord, const MazeSegment&); // Normal

	bool is_singular() const { return is_singular_; }
	ObVertex get_center() const;
private:
	ObVertex* corners_[4];
	void expand();

	std::vector<ObVertex*> bounaries_;
	size_t corner_pos_[4];
	double theta_;
	size_t neg_corner_;

	void build_boundary();
	void construct(MazeVertArray vs, double zcoord, const MazeSegment&);
	std::vector<std::pair<ObVertex*, ObVertex*> > neg_edges_, pos_edges_;
	bool is_singular_ = false;
	std::vector<ObVertex*> get_boundary_from_corner(size_t corder_idx, size_t n);
};

inline double cross2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
	return a.x() * b.y() - a.y() * b.x();
}
inline bool NClose(double value, double target)
{
	return std::abs(value - target) < kEpsilon;
}
inline double vector_angle(const Eigen::Vector2d& v0, const Eigen::Vector2d& v1)
{
	Eigen::Vector2d nv0 = v0.normalized();
	Eigen::Vector2d nv1 = v1.normalized();
#if 0
	std::cerr << __func__ << "\t" << v0.transpose() << "\t" << v1.transpose() << endl;
	std::cerr << __func__ << " normalized\t" << nv0.transpose() << "\t" << nv1.transpose() << endl;
#endif
	double dot = nv0.dot(nv1.normalized());
	double det = cross2d(nv1.normalized(), nv0.normalized());
	double ret = std::atan2(det, dot);
#if 0
	std::cerr << __func__ << " dot: " << dot << " cross: " << det << " atan2 " << ret << endl;
#endif
	return ret;
}

auto general_build_faces(const std::vector<ObVertex*>& lowline,
		const std::vector<ObVertex*>& highline,
		size_t start,
		size_t end
		)
{
	std::vector<ObFace> F;

	for(auto i = start; i < end; i++) {
		int ni = (i+1) % lowline.size();
		ObVertex* v00 = lowline[i];
		ObVertex* v01 = lowline[ni];
		ObVertex* v10 = highline[i];
		ObVertex* v11 = highline[ni];
		F.emplace_back(v00, v01, v10);
		F.emplace_back(v10, v01, v11);
	}
	return F;
}

auto noncircular_build_faces(const std::vector<ObVertex*>& lowline,
		const std::vector<ObVertex*>& highline)
{
	return general_build_faces(lowline, highline, 0, lowline.size() - 1);
}

auto circular_build_faces(const std::vector<ObVertex*>& lowline,
		const std::vector<ObVertex*>& highline)
{
	return general_build_faces(lowline, highline, 0, lowline.size());
}

auto build_faces_unbalanced(const std::vector<ObVertex*>& lowline,
		const std::vector<ObVertex*>& highline)
{
#if 0
	std::cerr << "UNB Sizes: " << lowline.size() << "  " << highline.size() << std::endl;
#endif
	std::vector<ObFace> F;
	size_t lb = 0, hb = 0;
	size_t lsize = lowline.size(), hsize = highline.size();
	while (lb < lsize && hb < hsize) {
		ObVertex *cl = lowline[lb], *ch = highline[hb];
		ObVertex *nv;
		if ((lb+1) * hsize < (hb+1) * lsize) {
			if (lb + 1 >= lsize)
				break;
			nv = lowline[lb + 1];
#if 0
			std::cerr << "UNB(llh): " << lb << lb + 1 << hb << std::endl;
#endif
			lb++;
		} else {
			if (hb + 1 >= hsize)
				break;
			nv = highline[hb + 1];
#if 0
			std::cerr << "UNB(lhh): " << lb << hb + 1 << hb << std::endl;
#endif
			hb++;
		}
#if 0
		std::cerr << cl << "  " << nv << "  " << ch << std::endl;
#endif
		F.emplace_back(cl, ch, nv);
	}
	return F;
}

void Obstacle::connect_parallelogram(LayerPolygon& low, LayerPolygon& high)
{
	std::vector<ObFace> F;

#if 1
	if (low.is_singular() || high.is_singular()) {
		std::vector<ObVertex*> lowline, highline;
		lowline = low.get_negative_strip();
		highline = high.get_negative_strip();
		F = noncircular_build_faces(lowline, highline);
		append_F(F);
#if 1
		lowline = low.get_positive_strip();
		highline = high.get_positive_strip();
		F = noncircular_build_faces(lowline, highline);
		append_F(F);
#endif

#if 1
		F = build_faces_unbalanced(low.get_left_side(), high.get_left_side());
		append_F(F);
		F = build_faces_unbalanced(low.get_right_side(), high.get_right_side());
		append_F(F);
#endif
		return ;
	}
#endif
	std::vector<ObVertex*> lowline, highline;
	lowline = low.get_boundary();
	highline = high.get_boundary();
	F = circular_build_faces(lowline, highline);
	append_F(F);
}

MazeSegment rotate(const MazeSegment& seg, const MazeVert& center, double theta)
{
	Eigen::Translation<double, 2> origin(-center);
	Eigen::Rotation2D<double> rotate(theta);
	Eigen::Translation<double, 2> back(center);
	Eigen::Transform<double, 2, Eigen::Affine> combined = back * rotate * origin;
	MazeSegment ret;
	ret.v0 = combined * seg.v0;
	ret.v1 = combined * seg.v1;
	return ret;
}

LayerPolygon build_parallelogram(const MazeSegment& wall,
		const MazeSegment& stick,
		const MazeVert& stick_center,
		double theta,
		bool enforce_singular = false)
{
	auto rs = rotate(stick, stick_center, theta);
#if 0
	std::cerr << theta << ":\t" << rs.v0.transpose() << std::endl << rs.v1.transpose() << std::endl;
#endif
	MazeVert ctrl_offset = stick_center - rs.v0;
	MazeVert stick_offset = rs.v1 - rs.v0;
	MazeVert v0 = wall.v0 + ctrl_offset;
	MazeVert v1 = wall.v0 - stick_offset + ctrl_offset;
	MazeVert v2 = wall.v1 - stick_offset + ctrl_offset;
	MazeVert v3 = wall.v1 + ctrl_offset;
	MazeVert wallv = wall.v1 - wall.v0;
	double cross = cross2d(stick_offset.normalized(), wallv.normalized());
#if 0
	std::cerr << " theta: " << theta << "\t cross: " << cross << endl;
#endif
	bool singular = enforce_singular || NClose(cross, 0.0);
	if (singular)
		return LayerPolygon({v0, v1, v2, v3}, theta, wall, cross);
	double dot = wallv.dot(stick_offset);
	MazeVertArray vs;
	if (cross > 0)
#if 0
		if (dot > 0)
			vs = {v1, v0, v3, v2};
		else
#endif
			vs = {v0, v3, v2, v1};
	else
#if 0
		if (dot > 0)
			vs = {v1, v2, v3, v0};
		else
#endif
			vs = {v0, v1, v2, v3};
	std::cerr << cross << '\t' << dot << std::endl;
	return LayerPolygon(vs, theta, wall);
}

Obstacle::Obstacle(Options& o)
	:opt(o)
{
}
void Obstacle::construct(const MazeSegment& wall,
		const MazeSegment& stick,
		const MazeVert& stick_center)
{
	std::vector<LayerPolygon> layers;
	double DeltaTheta = kDeltaTheta * (1 + 0.05 * drand48());
	double EndTheta = kEndTheta + opt.margin();
	double BaseTheta = kBaseTheta - opt.margin();
	for(int i = 0; true; i++) {
		double theta = DeltaTheta * i + BaseTheta;
		if (theta > EndTheta)
			theta = EndTheta;
		auto parallelogram = build_parallelogram(wall, stick, stick_center, theta);
#if 0
		auto rs = rotate(stick, stick_center, theta);
		parallelogram.translate(stick_center - rs.v0);
		parallelogram.expand();
#endif
		layers.emplace_back(parallelogram);
		if (theta == EndTheta)
			break;
	}
	// Add singularity layers
	double base_theta = vector_angle(stick.v1 - stick.v0, wall.v1 - wall.v0);
	double min_theta = layers.front().rotate_angle() + base_theta;
	double max_theta = layers.back().rotate_angle() + base_theta;
	for(int i = -1; i <= 4; i++) {
		if (i * M_PI < min_theta)
			continue;
		if (i * M_PI > max_theta)
			continue;
		double singular_angle = i * M_PI - base_theta;
#if 0
		std::cerr << "Planning to insert singular angle: " << singular_angle << endl;
#endif
		bool do_insert = true;
		for(const auto& layer : layers) {
			if (NClose(layer.rotate_angle(), singular_angle)) {
				do_insert = false; // Remove duplicated
				break;
			}
		}
		if (do_insert) {
			layers.emplace_back(build_parallelogram(
					wall,
					stick,
					stick_center,
					singular_angle
					)
				);
		}
	}
	std::sort(layers.begin(), layers.end(),
			[] (const LayerPolygon& lhs, const LayerPolygon& rhs) {
				return lhs.rotate_angle() < rhs.rotate_angle();
			}
		 );
	for(int i = 0; i < layers.size() - 1; i++) {
		connect_parallelogram(layers[i], layers[i+1]);
	}
	std::cerr << "Base theta: " << base_theta << std::endl;
	for(auto& layer: layers) {
		std::cerr << "Layer theta: " << layer.rotate_angle()
		          << " is singular: " << layer.is_singular()
		          << std::endl;
		append_V(layer.get_boundary());
	}
	seal(layers.front(), true);
	seal(layers.back(), false);
}

void Obstacle::build_VF(Eigen::MatrixXd& V, Eigen::MatrixXi& F)
{
	F.resize(F_.size(), 3);
	vimap_.clear();
	vi_ = 0;
	for(int i = 0; i < F_.size(); i++) {
		F(i,0) = locate(F_[i].verts[0]);
		F(i,1) = locate(F_[i].verts[1]);
		F(i,2) = locate(F_[i].verts[2]);
#if 0
		std::cerr << F.row(i) << std::endl;
		std::cerr << x << '\t' << y << '\t' << z << '\t' << std::endl;
		std::cerr << locate(F_[i].verts[0]) << '\t' << locate(F_[i].verts[1]) << '\t' << locate(F_[i].verts[2]) << '\t' << std::endl;
#endif
	}
	V.resize(vi_, 3);
	for(auto pair : vimap_) {
		V.row(pair.second) = *(pair.first);
	}
}

void Obstacle::append_F(const std::vector<ObFace>& NF)
{
	F_.insert(F_.end(), NF.begin(), NF.end());
#if 0
	std::cerr << "----appendF----" << std::endl;
	for(auto f: NF) {
		std::cerr << f.verts[0] << '\t' << f.verts[1] << '\t' << f.verts[2] << std::endl;
	}
	std::cerr << "----appendF----" << std::endl;
#endif
}

void Obstacle::append_V(const std::vector<ObVertex*>& NV)
{
	for(auto v : NV) {
		V_.emplace_back(v);
		//for(int i = 0; i < 3; i++)
		//std::cerr << '(' << v->transpose() << ")\t";
		//std::cerr << std::endl;
	}
}

LayerPolygon::LayerPolygon(MazeVertArray vs, double zcoord, const MazeSegment& wall, double cross)
{
	MazeVert wallX = wall.v1 - wall.v0;
	std::sort(vs.begin(), vs.end(), [wallX](const MazeVert& lhs, const MazeVert& rhs)
			{
				return wallX.dot(lhs) < wallX.dot(rhs);
			}
		 );
	MazeVert wallY = MazeVert(-wallX.y(), wallX.x()).normalized() * kWallEpsilon;
	construct({vs.front() - wallY,
			vs.back() - wallY,
			vs.back() + wallY,
			vs.front() + wallY},
			zcoord,
			wall);
	is_singular_ = true;

#if 0
	neg_edges_.emplace_back(corners_[0], corners_[1]);
	pos_edges_.emplace_back(corners_[2], corners_[3]);
#endif
	neg_corner_ = 0;
}

/*
 * Proper way: construct the convex
 */
LayerPolygon::LayerPolygon(MazeVertArray vs, double zcoord, const MazeSegment& wall)
{
	construct(vs, zcoord, wall);

	MazeVert wallX = wall.v1 - wall.v0;
	MazeVert wallY = MazeVert(-wallX.y(), wallX.x()).normalized();

	size_t minpidx = 0;
	double minpx = wallX.dot(vs[0] - wall.v0);
	double minpy = wallY.dot(vs[0] - wall.v0);
	for(size_t i = 1; i < 4; i++) {
		double px = wallX.dot(vs[i] - wall.v0);
		double py = wallY.dot(vs[i] - wall.v0);
		bool change = false;

		if (minpx > px)
			change = true;
		else if (minpx == px && minpy > py)
			change = true;

		if (change) {
			minpidx = i;
			minpx = px;
			minpy = py;
		}
	}
#if 0
	size_t idx = minpidx;
	size_t next;
	for(int i = 0; i < 2; i++, idx = next) {
		next = (idx + 1) % 4;
		neg_edges_.emplace_back(corners_[idx], corners_[next]);
	}
	for(int i = 0; i < 2; i++, idx = next) {
		next = (idx + 1) % 4;
		pos_edges_.emplace_back(corners_[idx], corners_[next]);
	}
#endif
	neg_corner_ = minpidx;
}

void LayerPolygon::construct(MazeVertArray vs, double zcoord, const MazeSegment& wall)
{
	MazeVertArray orthseq;
#if 0
	std::sort(vs.begin(), vs.end(), [](const MazeVert& lhs, const MazeVert& rhs)
			{
				return lhs.x() < rhs.x() ||
				(lhs.x() == rhs.x() && lhs.y() < rhs.y());
			}
		 );
	if (vs[1].y() < vs[2].y()) {
		orthseq = {vs[0], vs[1], vs[3], vs[2]};
	} else {
		orthseq = {vs[0], vs[2], vs[3], vs[1]};
	}
#else
	orthseq = {vs[0], vs[1], vs[2], vs[3]};
#endif

	for(int i = 0; i < 4; i++) {
		corners_[i] = new ObVertex(vs[i].x(), vs[i].y(), zcoord);
#if 0
		std::cerr << '(' << corners_[i]->transpose() << ")\t";
#endif
	}
	// Expand
	auto center = get_center();
	for(int i = 0; i < 4; i++) {
		*corners_[i] = (*corners_[i] - center) * kExpandRatio + center;
	}
	std::cerr << std::endl;

	theta_ = zcoord;
}

std::vector<ObVertex*>& LayerPolygon::get_boundary()
{
	if (bounaries_.empty()) {
		build_boundary();
	}
	return bounaries_;
}

void LayerPolygon::build_boundary()
{
	if (!is_singular()) {
		for(int i = 0; i < 4; i++) {
			corner_pos_[i] = bounaries_.size();
			bounaries_.emplace_back(corners_[i]);
			ObVertex v0(*corners_[i]), v1(*corners_[(i+1)%4]);
			for(int j = 1; j < kNInterp; j++) {
				double r = double(j)/kNInterp;
				bounaries_.emplace_back(new ObVertex(v0*(1-r)+v1*r));
			}
		}
		return ;
	}
	for(int i = 0; i < 4; i += 2) {
		corner_pos_[i] = bounaries_.size();
		bounaries_.emplace_back(corners_[i]);

		ObVertex v0(*corners_[i]), v1(*corners_[(i+1)%4]);
		for(int j = 1; j < 2 * kNInterp; j++) {
			double r = double(j)/(2 * kNInterp);
			bounaries_.emplace_back(new ObVertex(v0*(1-r)+v1*r));
		}
		corner_pos_[i+1] = bounaries_.size();
		bounaries_.emplace_back(corners_[i+1]);
	}
}

ObVertex LayerPolygon::get_center() const
{
	ObVertex ret;
	ret << 0.0, 0.0, 0.0;
	for(int i = 0; i < 4; i++)
		ret += *(corners_[i]);
	ret /= 4;
	return ret;
}

int Obstacle::locate(ObVertex* v)
{
	auto iter = vimap_.find(v);
	if (iter != vimap_.end()) {
#if 0
		std::cerr << v << ":\t" << iter->second << std::endl;
#endif
		return iter->second;
	}
#if 0
	std::cerr << v << ":\t" << vi_ << std::endl;
#endif
	return (vimap_[v] = vi_++);
}

void Obstacle::seal(LayerPolygon& layer, bool reverse)
{
	auto c = new ObVertex(layer.get_center());
	append_V({c});
	std::vector<ObVertex*> b = layer.get_boundary();
	if (reverse)
		std::reverse(b.begin(), b.end());
	b.emplace_back(b.front()); // Make a circular.
#if 0
	std::cerr << "---seal" << std::endl;
	for(auto p : b)
		std::cerr << p << std::endl;
	std::cerr << "---sealed" << std::endl;
#endif
	append_F(build_faces_unbalanced({c}, b));
}

std::vector<ObVertex*> LayerPolygon::get_boundary_from_corner(size_t corner_idx, size_t n)
{
	std::vector<ObVertex*> ret;
	auto iter = get_boundary().begin();
	iter += corner_pos_[corner_idx];
	for(size_t i = 0; i < n; i++) {
		ret.emplace_back(*iter);
		iter++;
		if (iter == get_boundary().end())
			iter = get_boundary().begin();
	}
	return ret;
}

std::vector<ObVertex*> LayerPolygon::get_negative_strip()
{
	return get_boundary_from_corner(neg_corner_, 2 * kNInterp + 1);

	auto start = get_boundary().begin();
	auto end = start + 2 * kNInterp + 1;
	std::vector<ObVertex*> ret(start, end);
	return ret;
}

std::vector<ObVertex*> LayerPolygon::get_positive_strip()
{
	return get_boundary_from_corner((neg_corner_ + 2)% 4, 2 * kNInterp + 1);

	auto start = get_boundary().begin() + 2 * kNInterp;
	auto end = get_boundary().end();
	if (is_singular())
		start += 1;
	std::vector<ObVertex*> ret(start, end);
	if (!is_singular()) 
		ret.emplace_back(get_boundary().front());
	return ret;
}

std::vector<ObVertex*> LayerPolygon::get_left_side()
{
	std::vector<ObVertex*> ret;
	if (is_singular()) {
		ret = {get_boundary().front(), get_boundary().back()};
	} else {
		ret = get_boundary_from_corner(neg_corner_, 1);
	}
	return ret;
}

std::vector<ObVertex*> LayerPolygon::get_right_side()
{
	std::vector<ObVertex*> ret;
	if (is_singular()) {
		ret = std::vector<ObVertex*>(get_boundary().begin() + 2 * kNInterp,
				get_boundary().begin() + 2 * kNInterp + 2);
		std::reverse(ret.begin(), ret.end());
	} else {
		ret = get_boundary_from_corner((neg_corner_ + 2)% 4, 1);
	}
	return ret;
}
