#ifndef GOCTREE_H
#define GOCTREE_H

/*
 * Generalized Octree
 *  An octree-like data structure for arbitrary dimension space.
 *  
 * Note: check https://octomap.github.io/ for more details.
 */

#include "goctree_child_policy.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

template<int ND, typename FLOAT = double> // FIXME: use typename ChildPolicy = GOcTreeDefaultChildPolicy<ND, FLOAT>
class GOcTreeNode {
public:
	typedef Eigen::Matrix<FLOAT, ND, 1> Coord;
	typedef std::bitset<ND> CubeIndex;
	enum CubeState {
		kCubeUncertain,
		kCubeFree,
		kCubeFull,
	};

private:
	struct ChildPolicy {
		typedef typename std::vector<std::unique_ptr<GOcTreeNode>> ChildType;

		static void initChildren(ChildType& children)
		{
			children.resize(ND);
		}

		static void assignCube(ChildType& children, unsigned long offset, std::unique_ptr<GOcTreeNode>&& node)
		{
			children[offset] = node;
		}

		static GOcTreeNode* accessCube(ChildType& children, unsigned long offset)
		{
			children[offset].get();
		}
	};
	ChildType children_;

	Coord mins_, maxs_;
	Coord median_;
	int depth_;
	CubeState state_;
public:
	GOcTreeNode(const Coord& mins, const Coord& maxs, int depth)
		:mins_(mins), maxs_(maxs), depth_(depth), state_(kCubeUncertain)
	{
		median_ = FLOAT(0.5) * mins + FLOAT(0.5) * maxs;
		ChildPolicy::initChildren(children_);
	}

	void getBV(Coord& mins, Coord& maxs) const
	{
		mins = mins_;
		maxs = maxs_;
	}

	void getCubeBV(const CubeIndex& ci, Coord& mins, Coord& maxs) const
	{
		for (int i = 0; i < ND; i++) {
			if (!ci[i]) {
				// false: lower half
				mins(i) = mins_(i);
				maxs(i) = median_(i);
			} else {
				// true: upper half
				mins(i) = median_(i);
				maxs(i) = maxs_(i);
			}
		}
	}

	int getDepth() const { return depth_; }

	CubeIndex locateCube(const Coord& coord) const
	{
		CubeIndex ret;
		for (int i = 0; i < ND; i++) {
			ret[i] = (coord(i) > median_(i));
		}
		return ret;
	}
	
	void setState(CubeState s) { state_ = s; }
	CubeState getState() const { return state_; }

	void expandCube(const CubeIndex& ci)
	{
		unsigned long index = ci.to_ulong();
		Coord mins, maxs;
		getCubeBV(ci, mins, maxs);

		std::unique_ptr<GOcTreeNode> node(new GOcTreeNode(mins, maxs, depth_ + 1));
		ChildPolicy::assignCube(children_, index, std::move(node));
	}

	GOcTreeNode* getCube(const CubeIndex& ci)
	{
		unsigned long index = ci.to_ulong();
		auto ret = ChildPolicy::accessCube(children_, index);
		if (!ret)
			expandCube(ci);
		return ChildPolicy::accessCube(children_, index);
	}
};

#endif
