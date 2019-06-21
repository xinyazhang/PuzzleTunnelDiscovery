#include <vector>
#include <iostream>
#include <unordered_set>
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/geometry/shape/box.h>

#include "RectReal.h"
#include "FreeRectangleManager.h"

namespace rbp {

/// @return True if the free node was split.
bool SplitFreeNode(const Rect& freeNode, const Rect &usedNode, std::vector<Rect>& newNodes)
{
	// Test with SAT if the rectangles even intersect.
	if (usedNode.x >= freeNode.x + freeNode.width || usedNode.x + usedNode.width <= freeNode.x ||
		usedNode.y >= freeNode.y + freeNode.height || usedNode.y + usedNode.height <= freeNode.y)
		return false;

	if (usedNode.x < freeNode.x + freeNode.width && usedNode.x + usedNode.width > freeNode.x)
	{
		// New node at the top side of the used node.
		if (usedNode.y > freeNode.y && usedNode.y < freeNode.y + freeNode.height)
		{
			Rect newNode = freeNode;
			newNode.height = usedNode.y - newNode.y;

			newNodes.emplace_back(newNode);
		}

		// New node at the bottom side of the used node.
		if (usedNode.y + usedNode.height < freeNode.y + freeNode.height)
		{
			Rect newNode = freeNode;
			newNode.y = usedNode.y + usedNode.height;
			newNode.height = freeNode.y + freeNode.height - (usedNode.y + usedNode.height);
			newNodes.emplace_back(newNode);
		}
	}

	if (usedNode.y < freeNode.y + freeNode.height && usedNode.y + usedNode.height > freeNode.y)
	{
		// New node at the left side of the used node.
		if (usedNode.x > freeNode.x && usedNode.x < freeNode.x + freeNode.width)
		{
			Rect newNode = freeNode;
			newNode.width = usedNode.x - newNode.x;

			newNodes.emplace_back(newNode);
		}

		// New node at the right side of the used node.
		if (usedNode.x + usedNode.width < freeNode.x + freeNode.width)
		{
			Rect newNode = freeNode;
			newNode.x = usedNode.x + usedNode.width;
			newNode.width = freeNode.x + freeNode.width - (usedNode.x + usedNode.width);

			newNodes.emplace_back(newNode);
		}
	}
	// std::cout << "\n\tSplit freeNode " << freeNode.width << " " << freeNode.height << " at " << freeNode.width << " " << freeNode.y << std::endl;

	return true;
}

#if FRM_HAS_REFERENCE
template<bool verbose>
void PruneFreeList(std::vector<Rect>& freeRectangles)
{
	/*
	///  Would be nice to do something like this, to avoid a Theta(n^2) loop through each pair.
	///  But unfortunately it doesn't quite cut it, since we also want to detect containment.
	///  Perhaps there's another way to do this faster than Theta(n^2).

	if (freeRectangles.size() > 0)
		clb::sort::QuickSort(&freeRectangles[0], freeRectangles.size(), NodeSortCmp);

	for(size_t i = 0; i < freeRectangles.size()-1; ++i)
		if (freeRectangles[i].x == freeRectangles[i+1].x &&
		    freeRectangles[i].y == freeRectangles[i+1].y &&
		    freeRectangles[i].width == freeRectangles[i+1].width &&
		    freeRectangles[i].height == freeRectangles[i+1].height)
		{
			freeRectangles.erase(freeRectangles.begin() + i);
			--i;
		}
	*/
	if (verbose) {
		std::cerr << "=====Prune begin=====" << std::endl;
		std::cerr << "List size: " << freeRectangles.size() << std::endl;
	}

	/// Go through each pair and remove any rectangle that is redundant.
	for(size_t i = 0; i < freeRectangles.size(); ++i) {
		for(size_t j = i+1; j < freeRectangles.size(); ++j)
		{
			if (IsContainedIn(freeRectangles[i], freeRectangles[j]))
			{
				if (verbose)
					std::cerr << "Prune box " << i << " inside " << j << std::endl;
				freeRectangles.erase(freeRectangles.begin()+i);
				--i;
				break;
			}
			if (IsContainedIn(freeRectangles[j], freeRectangles[i]))
			{
				if (verbose)
					std::cerr << "Prune box " << j << " inside " << i << std::endl;
				freeRectangles.erase(freeRectangles.begin()+j);
				--j;
			}
		}
	}
	if (verbose)
		std::cerr << "=====Prune finish=====" << std::endl;
}

template<bool verbose = true>
long
PruneFreeListCheck(const std::vector<std::unique_ptr<Rect>>& freeRectangles)
{
	long total = 0;
	for (size_t i = 0; i < freeRectangles.size(); ++i) {
		for (size_t j = i+1; j < freeRectangles.size(); ++j)
		{
			if (IsContainedIn(*freeRectangles[i], *freeRectangles[j]))
			{
				if (verbose)
					std::cerr << "Need to Prune box " << i << " inside " << j << std::endl;
				total += 1;
				break;
			}
			if (IsContainedIn(*freeRectangles[j], *freeRectangles[i]))
			{
				if (verbose)
					std::cerr << "Need to Prune box " << i << " inside " << j << std::endl;
				total += 1;
			}
		}
	}
	if (verbose)
		std::cerr << "Need to Prune " << total << " rectangles" << std::endl;
}
#endif

struct FreeRectangleManager::InternalData {
	using Scalar = double;
	using Vector3 = fcl::Vector3<Scalar>;
	using BoxGeom = fcl::Box<Scalar>;
	using Translation3 = fcl::Translation3<Scalar>;
	using Transform3 = fcl::Transform3<Scalar>;
	using CollisionObject = fcl::CollisionObject<Scalar>;

	// fcl::DynamicAABBTreeCollisionManager_Array<Scalar> aabb;
	fcl::DynamicAABBTreeCollisionManager<Scalar> aabb;
	std::vector<std::unique_ptr<CollisionObject>> box_instances;
	std::vector<std::unique_ptr<Rect>> free_rectangles;

	void addRect(const Rect& rect)
	{
		Rect *n = new Rect;
		*n = rect;
		free_rectangles.emplace_back(n);

		auto box = std::make_shared<BoxGeom>(rect.width, rect.height, 1.0);
		auto cobj = new CollisionObject(box, Transform3 {Translation3(Vector3(rect.x, rect.y, 0.0))});
		cobj->setUserData(n);
		box_instances.emplace_back(cobj);
		aabb.registerObject(cobj);
	}
};


FreeRectangleManager::FreeRectangleManager(const Rect& root)
	:d_(new InternalData)
{
	d_->aabb.tree_init_level = 2;
	d_->addRect(root);
	d_->aabb.setup();
}

FreeRectangleManager::~FreeRectangleManager()
{
}

bool
collect_prune_list(fcl::CollisionObject<double>* obj1,
                   fcl::CollisionObject<double>* obj2,
                   void *cookie)
{
	auto& prune_list = *reinterpret_cast<std::unordered_set<Rect*>*>(cookie);
	auto up1 = obj1->getUserData();
	auto up2 = obj2->getUserData();
	if (!up1 || !up2)
		return false;
	auto& rect1 = *reinterpret_cast<Rect*>(up1);
	auto& rect2 = *reinterpret_cast<Rect*>(up2);

	if (IsContainedIn(rect1, rect2)) {
		prune_list.emplace(&rect1);
	} else if (IsContainedIn(rect2, rect1)) {
		prune_list.emplace(&rect2);
	}
	return false;
}

void
FreeRectangleManager::PlaceRect(const Rect &node)
{
	// std::cerr << "***** First Prune *****" << std::endl;
	// PruneFreeList(freeRectangles);
	std::vector<Rect> newNodes;
	auto box_iter = d_->box_instances.begin();
	auto iter = d_->free_rectangles.begin();
	for (; iter != d_->free_rectangles.end();) {
		// std::cerr << "Offset " << iter - freeRectangles.begin() << std::endl;
		if (SplitFreeNode(**iter, node, newNodes)) {
			iter = d_->free_rectangles.erase(iter);
			d_->aabb.unregisterObject(box_iter->get());
			box_iter = d_->box_instances.erase(box_iter);
		} else {
			++iter;
			++box_iter;
		}
		// std::cerr << "Size " << freeRectangles.size() << std::endl;
	}
	if (!newNodes.empty()) {
		d_->free_rectangles.reserve(d_->free_rectangles.size() + newNodes.size()); 
		for (const auto& rect : newNodes) {
			d_->addRect(rect);
		}
	}
#if FRM_HAS_REFERENCE
	// std::cerr << "***** Second Prune *****" << std::endl;
	// PruneFreeList<false>(d_->free_rectangles);
	PruneFreeListCheck(d_->free_rectangles);
#endif
	d_->aabb.update();
	std::unordered_set<Rect*> prune_list;
	d_->aabb.collide(&prune_list, collect_prune_list);
#if FRM_HAS_REFERENCE
	std::cerr << "Found " << prune_list.size() << " items to prune" << std::endl;
#endif

	iter = d_->free_rectangles.begin();
	box_iter = d_->box_instances.begin();
	for (; box_iter != d_->box_instances.end();) {
		if (prune_list.find(iter->get()) != prune_list.end()) {
			iter = d_->free_rectangles.erase(iter);
			d_->aabb.unregisterObject(box_iter->get());
			box_iter = d_->box_instances.erase(box_iter);
		} else {
			++iter;
			++box_iter;
		}
	}
}

size_t
FreeRectangleManager::size() const
{
	return d_->free_rectangles.size();
}

const Rect&
FreeRectangleManager::getFree(size_t off) const
{
	return *d_->free_rectangles[off];
}

}
