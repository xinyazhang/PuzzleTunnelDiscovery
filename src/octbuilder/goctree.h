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
#include <bitset>
#include <set>
#include <ostream>

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
		typedef std::vector<std::unique_ptr<GOcTreeNode>> ChildType;

		static void initChildren(ChildType& children)
		{
			//children.resize(1 << ND);
			//for (auto& child: children)
			//	child.reset();
		}

		static void assignCube(ChildType& children, unsigned long offset, std::unique_ptr<GOcTreeNode>& node)
		{
			if (children.empty())
				children.resize(1 << ND);
			children[offset].swap(node);
		}

		static GOcTreeNode* accessCube(ChildType& children, unsigned long offset)
		{
			if (children.empty())
				return nullptr;
			return children[offset].get();
		}

		static bool isEmpty(const ChildType& children)
		{
			return children.empty();
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
	bool isLeaf() const { return ChildPolicy::isEmpty(children_); }
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
		Coord delta { Coord::Zero() };
		delta(dimension) = FLOAT(direction) * (from->maxs_(dimension) - from->mins_(dimension));
		Coord neighCenter = Space::transist(center, delta);
		std::cerr << "getNeighbor at " << neighCenter.transpose() << " from " << center.transpose() << " and delta " << delta.transpose() << std::endl;
		auto current = root;
		while (true) {
#if 1 // VERBOSE
			std::cerr << "\tProbing " << *current << std::endl;
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

	static std::vector<GOcTreeNode*>
	getBoundaryDescendant(GOcTreeNode* from, int dimension, int direction)
	{
		if (from->isLeaf())
			return {from};
		bool expbit = direction < 0 ? 0 : 1;
		std::vector<GOcTreeNode*> ret;
		bool debug = false;
		if (fabs(from->getMedian()(0) - (-2.5)) < 1e-3)
			if (fabs(from->getMedian()(1) - (-2.5)) < 1e-3)
				debug = true;
		if (debug) {
			std::cerr << "= Trying to get descendant from " << *from
				  << "\tdim: " << dimension << "\tdirection: " << direction
				  << std::endl;
		}
		// FIXME: better efficiency in iterating children
		for (unsigned long index = 0; index < (1 << ND); index++) {
			CubeIndex ci(index);
			if (ci[dimension] != expbit)
				continue;
			auto descendant = from->tryCube(ci);
			if (!descendant)
				continue;
			auto accum = getBoundaryDescendant(descendant, dimension, direction);
			if (debug) {
				std::cerr << "\t\t" << __func__ << " ci: " << ci
					  << "\t\t" << *descendant
					  << "\t\taccum size: " << accum.size()
					  << std::endl;
			}
			ret.insert(std::end(ret), std::begin(accum), std::end(accum));
		}
		return ret;
	}

	static void setAdjacency(GOcTreeNode *lhs, GOcTreeNode *rhs)
	{
		if (lhs == rhs)
			return ;
		lhs->adj_.insert(rhs);
		rhs->adj_.insert(lhs);
	}

	const std::set<GOcTreeNode*>& getAdjacency() const { return adj_; }
private:
	std::set<GOcTreeNode*> adj_;
};

template<int ND, typename FLOAT = double, typename UserDefinedAtrribute>
std::ostream& operator<<(std::ostream& fout, const GOcTreeNode<ND, FLOAT, UserDefinedAtrribute>& node)
{
	fout << "\tcenter: " << node.getMedian().transpose()
	     << "\tmins: " << node.getMins().transpose()
	     << "\tmaxs: " << node.getMaxs().transpose()
	     << "\tdepth: " << node.getDepth()
	     << "\tstate: " << node.getState();
	return fout;
}

#endif
