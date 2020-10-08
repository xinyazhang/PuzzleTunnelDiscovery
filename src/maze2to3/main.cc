/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <functional>
#include <map>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <Eigen/Geometry> // For cross product
#include <strman/suffix.h>
#include <memory>
#include <cassert>

using std::cout;
using std::cerr;
using std::endl;
using Eigen::Vector3i;
using Eigen::Vector2i;

const Vector2i kDiff[] = {
	{ 0,-1}, // LEFT
	{ 0, 1}, // RIGHT
	{-1, 0}, // UP
	{ 1, 0}, // DOWN
};
static constexpr int kDir = 4;
enum {
	kL = (1 << 0),
	kR = (1 << 1),
	kU = (1 << 2),
	kD = (1 << 3),
	kF = (1 << kDir)
};
const Vector3i posz = { 0, 0, 1 };
const Vector3i negz = { 0, 0, 1 };

void usage()
{
	cerr <<
R"xxx(
Usage: maze2to3 <.maze input file> <.obj output file>
Unit test: maze2to3 --sc
)xxx";
}

void
read_maze(std::istream&& fin,
          Eigen::MatrixXi& mc)
{
	int notused;
	fin >> notused >> notused;
	int row, col;
	fin >> row >> col;
	mc.resize(row, col);
	for(int i = 0; i < row; i++)
		for(int j = 0; j < col; j++)
			fin >> mc(i,j);
}

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
	return std::max(lower, std::min(n, upper));
}

void
gen_boundary(const Eigen::MatrixXi& mc,
             Eigen::MatrixXi& binfo)
{
	binfo.setZero(mc.rows(), mc.cols());

	auto cf = [&mc](Vector2i& v) {
		v(0) = clip<int>(v(0), 0, mc.rows());
		v(1) = clip<int>(v(1), 0, mc.cols());
	};
	for(int i = 0; i < mc.rows(); i++) {
		for(int j = 0; j < mc.cols(); j++) {
			if (mc(i,j) == 0)
				continue;
			Vector2i o(i, j);
			binfo(i, j) |= (1 << (kDir));
			for(int dir = 0; dir < kDir; dir++) {
				Vector2i p = o + kDiff[dir];
				cf(p);
				if (mc(p(0), p(1)) != mc(i,j))
					binfo(i,j) |= (1 << dir);
			}
		}
	}
}

#if 0
bool operator<(const Vector3i& a, const Vector3i& b)
{
	assert(a.size() == b.size());
	for(size_t i = 0; i < a.size(); ++i) {
		if (a[i]<b[i]) return true;
		if (a[i]>b[i]) return false;
	}
	return false;
}
#else
namespace std
{
template<> struct less<::Eigen::Vector3i>
{
	bool operator() (const ::Eigen::Vector3i& a, const ::Eigen::Vector3i& b) const
	{
		assert(a.size() == b.size());
		for (size_t i = 0; i < a.size(); ++i) {
			if (a[i] < b[i]) return true;
			if (a[i] > b[i]) return false;
		}
		return false;
	}
};
}
#endif

//
// IntMeshSanitizer: Takes 3D Int Square
//
class IntMeshSanitizer {
	struct IndexWrapper {
		IndexWrapper()
			:index(new int)
		{
			*index = -1;
		}
		std::shared_ptr<int> index;
	};
	struct Triangle {
		Triangle(IndexWrapper& v0,
		         IndexWrapper& v1,
		         IndexWrapper& v2)
		{
			this->v0 = v0;
			this->v1 = v1;
			this->v2 = v2;
		}
		IndexWrapper v0;
		IndexWrapper v1;
		IndexWrapper v2;
	};
	typedef Eigen::Matrix<int, 3, -1> ExpanderMatrix;
	struct OrthoExpander {
		OrthoExpander()
		{
			static constexpr int kExpanderDim = 3;
			Vector3i n;
			for (int d = 0; d < kExpanderDim; d++) {
				n.setZero();
				for (auto sign : {-1, 1}) {
					n(d) = sign;
					fillExpander(n);
				}
			}
		}

		static OrthoExpander* getInstance()
		{
			static OrthoExpander inst;
			return &inst;
		}

		// IN:
		//   n: normal
		// OUT:
		//   None
		//
		// Effect:
		//   Correct expander was added to expander_[n]
		//
		// Term Elaboration:
		//   Expander, a set of vectors that forms axis-aligned
		//   surface that is orthogonal to n from a given origin
		//   point o.
		//   Note for simplicity this surface is a square with side
		//   lenght = 2, the origin point o is at the center of this
		//   square.
		void fillExpander(const Vector3i& n)
		{
			std::vector<Vector3i> ortho;
			fillOrtho(n, ortho);
			assert(ortho.size() == 4);
			std::vector<Vector3i> ex;
			// Pick up corner vectors to ex
			for (size_t i = 0; i < ortho.size(); i++) {
				const auto& ax1 = ortho[i];
				for (size_t j = i + 1; j < ortho.size(); j++) {
					const auto& ax2 = ortho[j];
					// Skip non-orthogonal pairs
					if (ax1.dot(ax2) != 0)
						continue;
					ex.emplace_back(ax1 + ax2);
				}
			}
			// Re-order to honor right hand rule
			// Algo: start with any vertex, and move one from ex
			//       to sorted_ex in each iteration.
			std::vector<Vector3i> sorted_ex;
			sorted_ex.emplace_back(ex.front());
			ex.erase(ex.begin());
			while (!ex.empty()) {
				Vector3i start = sorted_ex.back();
				for (auto iter = ex.begin();
				     iter != ex.end();
				     iter++) {
					Vector3i edge = *iter - start;
					// Exclude diagonal edges
					if (edge.squaredNorm() != 4)
						continue;
					// Exclude clock-wise (violate right hand rule)
					// pairs
					Vector3i cross = start.cross(*iter);
					if (cross.dot(n) < 0)
						continue;
					sorted_ex.emplace_back(*iter);
					iter = ex.erase(iter);
					break;
				}
			}
			ex = std::move(sorted_ex);

			ExpanderMatrix mex;
			mex.resize(3, ex.size());
			for (size_t i = 0; i < ex.size(); i++) {
				mex.col(i) = ex[i];
			}
			expander_[n] = mex;
		}

		void fillOrtho(const Vector3i& n,
		               std::vector<Vector3i>& ortho)
		{
			static constexpr int kExpanderDim = 3;
			Vector3i ax;
			for (int d = 0; d < kExpanderDim; d++) {
				ax.setZero();
				for (auto sign : {-1, 1}) {
					ax(d) = sign;
					// Exclude non-orthogonal axes.
					if (ax.dot(n) != 0)
						continue;
					ortho.emplace_back(ax);
				}
			}
		}

		ExpanderMatrix getExpand(const Vector3i& n) const
		{
			auto iter = expander_.find(n);
			if (iter == expander_.end()) {
				std::ostringstream err;
				err << __func__ << " cannot find expander for normal " << n;
				throw std::runtime_error(err.str());
			}
			return iter->second;
		}

		std::map<Vector3i, ExpanderMatrix> expander_;
	};
public:
	IntMeshSanitizer()
	{
	}

	void
	addSurface(const Vector3i& o,
	           const Vector3i& n)
	{
		auto expander = OrthoExpander::getInstance();
		auto ex = expander->getExpand(n);
		// Vector3i r0 = o + ex.col(0);
#if 1
		addSurface(o + ex.col(0),
		           o + ex.col(1),
		           o + ex.col(2),
		           o + ex.col(3));
#endif
	}

	// IN
	//      Four vertices in counter clockwise order w.r.t -normal view
	//      direction
	void
	addSurface(const Vector3i& tl,
	           const Vector3i& tr,
	           const Vector3i& br,
	           const Vector3i& bl)
	{
		addTriangle(tl, tr, br);
		addTriangle(tl, br, bl);
	}

	void
	addTriangle(const Vector3i& v0,
	            const Vector3i& v1,
	            const Vector3i& v2)
	{
		tris_.emplace_back(cache_[v0],
		                   cache_[v1],
		                   cache_[v2]);
	}

	void serialize(Eigen::MatrixXd& V,
	               Eigen::MatrixXi& F)
	{
		int vi = 0;
		// Fill V and assign indices
		V.resize(cache_.size(), 3);
		for (const auto& item: cache_) {
			*(item.second.index) = vi;
			V.row(vi) = item.first.cast<double>();
			vi++;
		}
		int fi = 0;
		F.resize(tris_.size(), 3);
		for (const auto& tri: tris_) {
			F(fi, 0) = *(tri.v0.index);
			F(fi, 1) = *(tri.v1.index);
			F(fi, 2) = *(tri.v2.index);
			fi++;
		}
	}
protected:
	std::map<Vector3i, IndexWrapper> cache_;
	std::vector<Triangle> tris_;
};

void
gen_mesh(const Eigen::MatrixXi& binfo,
         Eigen::MatrixXd& V,
         Eigen::MatrixXi& F)
{
	IntMeshSanitizer ims;
	for(int i = 0; i < binfo.rows(); i++) {
		for(int j = 0; j < binfo.cols(); j++) {
			if (binfo(i,j) == 0)
				continue;
			Vector3i o(2*i, 2*j, 0);
			ims.addSurface(o + posz, posz);
			ims.addSurface(o + negz, negz);
			for(int dir = 0; dir < kDir; dir++) {
				if ((binfo(i, j) & (1 << dir)) == 0)
					continue;
				const auto& diff = kDiff[dir];
				Vector3i n(diff(0), diff(1), 0);
				Vector3i so = o + n; // Surface center
				ims.addSurface(so, n);
			}
		}
	}
	ims.serialize(V, F);
}

void sancheck();

int main(int argc, char* argv[])
{
	if (argc >= 2 && argv[1] == std::string("--sc")) {
		sancheck();
		return 0;
	}
	if (argc < 3) {
		usage();
		return -1;
	}
	if (!strman::has_suffix(argv[1], ".maze")) {
		cerr << argv[1] << " does not have suffix .maze" << endl;
		usage();
		return -1;
	}
	if (!strman::has_suffix(argv[2], ".obj")) {
		cerr << argv[2] << " does not have suffix .obj" << endl;
		usage();
		return -1;
	}
	// Read maze
	Eigen::MatrixXi mc;
	read_maze(std::ifstream(argv[1]),
	          mc);

	// Convert maze to boundary info
	Eigen::MatrixXi binfo; // horizontal boundary description.
	gen_boundary(mc, binfo);

	// Convert maze to boundary info
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	gen_mesh(binfo, V, F);

	igl::writeOBJ(argv[2], V, F);
	return 0;
}

#include <cstdlib>

#define VERIFY(X, Y)                                                    \
	do {                                                            \
		bool ret = (X);                                         \
		auto dval = (Y);                                        \
		if (ret == false) {                                     \
			std::cerr << #X << " failed\n";                 \
			std::cerr << "Diagnose info " << #Y " = " << dval << std::endl; \
			std::abort();                                   \
		}                                                       \
	} while (0)

void sancheck()
{
	Eigen::MatrixXi mc;
	mc.setZero(3,3);
	mc(1,1) = 1;
	Eigen::MatrixXi binfo;
	gen_boundary(mc, binfo);
	VERIFY(binfo(1,1) == (kL | kR | kU | kD | kF), binfo(1,1));
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	gen_mesh(binfo, V, F);

	igl::writeOBJ("sc.obj", V, F);
}
