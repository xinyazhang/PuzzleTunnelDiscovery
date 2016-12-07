#ifndef GOCTREE_H
#define GOCTREE_H

/*
 * Generalized Octree
 *  An octree-like data structure for arbitrary dimension space.
 *  
 * Note: check https://octomap.github.io/ for more details.
 */

#include <Eigen/Core>
#include <memory>
#include <vector>

struct GOcTreeNodeNoAttribute {
};

template<int ND, typename FLOAT = double, typename UserDefinedAtrribute = GOcTreeNodeNoAttribute> // FIXME: use typename ChildPolicy = GOcTreeDefaultChildPolicy<ND, FLOAT>
class GOcTreeNode : public UserDefinedAtrribute {
public:
	typedef Eigen::Matrix<FLOAT, ND, 1> Coord;
	typedef std::bitset<ND> CubeIndex;
	enum CubeState {
		kCubeUncertain,
		kCubeUncertainPending,
		kCubeMixed,
		kCubeFree,
		kCubeFull,
	};

private:
	struct ChildPolicy {
		typedef std::unique_ptr<GOcTreeNode> ChildType[1<<ND];

		static void initChildren(ChildType& children)
		{
			//children.resize(1 << ND);
			for (auto& child: children)
				child.reset();
		}

		static void assignCube(ChildType& children, unsigned long offset, std::unique_ptr<GOcTreeNode>& node)
		{
			children[offset].swap(node);
		}

		static GOcTreeNode* accessCube(ChildType& children, unsigned long offset)
		{
			return children[offset].get();
		}
	};
	typename ChildPolicy::ChildType children_;

	GOcTreeNode* parent_;
	Coord mins_, maxs_;
	Coord median_;
	CubeState state_;
	CubeIndex parent_to_this_ci_;
	unsigned char depth_;
public:
	GOcTreeNode(GOcTreeNode* parent, CubeIndex ci, const Coord& mins, const Coord& maxs)
		:parent_(parent),
		 mins_(mins),
		 maxs_(maxs),
		 state_(kCubeUncertain),
		 parent_to_this_ci_(ci)
	{
		median_ = FLOAT(0.5) * mins + FLOAT(0.5) * maxs;
		ChildPolicy::initChildren(children_);
	}

	void getBV(Coord& mins, Coord& maxs) const
	{
		mins = mins_;
		maxs = maxs_;
	}

	Coord getMins() const { return mins_; }
	Coord getMaxs() const { return maxs_; }

	double getVolume() const
	{
		Coord vol = maxs_ - mins_;
		double ret = 1.0;
		for (int i = 0; i < ND; i++)
			ret *= vol(i);
		return ret;
	}

	Coord getMedian() const { return median_; }

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

	bool isContaining(const Coord& coord)
	{
		for (int i = 0; i < ND; i++) {
			if (mins_(i) > coord(i) || maxs_(i) < coord(i))
				return false;
		}
		return true;
	}

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
	bool isLeaf() const { return state_ == kCubeFree || state_ == kCubeFull; }
	unsigned getDepth() const { return unsigned(depth_); }

	void expandCube(const CubeIndex& ci)
	{
		unsigned long index = ci.to_ulong();
		Coord mins, maxs;
		getCubeBV(ci, mins, maxs);

		std::unique_ptr<GOcTreeNode> node(new GOcTreeNode(this, ci, mins, maxs));
		node->depth_ = depth_ + 1;
		//std::cerr << __func__ << ": " << node.get() << std::endl;
		ChildPolicy::assignCube(children_, index, node);
	}

	GOcTreeNode* tryCube(const CubeIndex& ci)
	{
		unsigned long index = ci.to_ulong();
		return ChildPolicy::accessCube(children_, index);
	}

	GOcTreeNode* getCube(const CubeIndex& ci)
	{
		unsigned long index = ci.to_ulong();
		auto ret = ChildPolicy::accessCube(children_, index);
		if (!ret)
			expandCube(ci);
		return ChildPolicy::accessCube(children_, index);
	}

	static GOcTreeNode* makeRoot(const Coord& mins, const Coord& maxs)
	{
		auto ret = new GOcTreeNode(nullptr, {}, mins, maxs);
		ret->depth_ = 0;
		return ret;
	}

	template<typename Space>
	static GOcTreeNode*
	getNeighbor(GOcTreeNode* root, GOcTreeNode* from, int dimension, int direction)
	{
		Coord center = from->getMedian();
		Coord delta;
		delta(dimension) = FLOAT(direction) * (from->maxs_(dimension) - from->mins_(dimension));
		Coord neighCenter = Space::transist(center, delta);
		auto current = root;
		while (true) {
#if VERBOSE
			std::cerr << "Probing neighbor (" << current->getMins().transpose()
				<< ")\t(" << current->getMaxs().transpose() << ")" << std::endl;
#endif
			auto ci = current->locateCube(neighCenter);
			auto next = current->tryCube(ci);
			if (!next || next->getDepth() > from->getDepth())
				break;
			current = next;
		}
#if VERBOSE
		std::cerr << __func__ << " returns " << current << std::endl;
#endif
		return current;
	}

	template<typename Space>
	static std::vector<GOcTreeNode*>
	getContactCubes(GOcTreeNode* root,
			GOcTreeNode* from,
			int dimension,
			int direction,
			Space
			)
	{
		auto neighbor = getNeighbor<Space>(root, from, dimension, direction);
		if (neighbor->isLeaf() || neighbor->getState() == kCubeUncertain)
			return {neighbor};
		return getBoundaryDescendant(neighbor, dimension, -direction);
	}

	// TODO: Fill the skeleton.
	static std::vector<GOcTreeNode*>
	getBoundaryDescendant(GOcTreeNode* from, int dimension, int direction)
	{
		if (from->isLeaf())
			return {};
		bool expbit = direction < 0 ? 0 : 1;
		std::vector<GOcTreeNode*> ret;
		// FIXME: better efficiency in iterating children
		for (unsigned long index = 0; index < (1 << ND); index++) {
			CubeIndex ci(index);
			if (ci[dimension] != expbit)
				continue;
			auto descendant = from->tryCube(ci);
			if (!descendant)
				continue;
			auto accum = getBoundaryDescendant(descendant, dimension, direction);
			ret.insert(std::end(ret), std::begin(accum), std::end(accum));
		}
		return ret;
	}
};

#endif
