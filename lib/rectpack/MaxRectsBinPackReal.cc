/** @file MaxRectsBinPack.cpp
	@author Jukka Jylänki

	@brief Implements different bin packer algorithms that use the MAXRECTS data structure.

	This work is released to Public Domain, do whatever you want with it.
*/
#include <algorithm>
#include <utility>
#include <iostream>
#include <limits>

#include <cassert>
#include <cstring>
#include <cmath>

#include "MaxRectsBinPackReal.h"

namespace rbp {

using namespace std;

MaxRectsBinPack::MaxRectsBinPack()
:binWidth(0),
binHeight(0)
{
}

MaxRectsBinPack::MaxRectsBinPack(double width, double height, bool allowFlip)
{
	Init(width, height, allowFlip);
}

void MaxRectsBinPack::Init(double width, double height, bool allowFlip)
{
	binAllowFlip = allowFlip;
	binWidth = width;
	binHeight = height;

	Rect n;
	n.x = 0;
	n.y = 0;
	n.width = width;
	n.height = height;

	usedRectangles.clear();

	freeRectangles.clear();
	freeRectangles.push_back(n);
}

Rect MaxRectsBinPack::Insert(double width, double height, FreeRectChoiceHeuristic method, void *cookie)
{
	Rect newNode;
	// Unused in this function. We don't need to know the score after finding the position.
	double score1 = std::numeric_limits<double>::max();
	double score2 = std::numeric_limits<double>::max();
	switch(method)
	{
		case RectBestShortSideFit: newNode = FindPositionForNewNodeBestShortSideFit(width, height, score1, score2); break;
		case RectBottomLeftRule: newNode = FindPositionForNewNodeBottomLeft(width, height, score1, score2); break;
		case RectContactPointRule: newNode = FindPositionForNewNodeContactPoint(width, height, score1); break;
		case RectBestLongSideFit: newNode = FindPositionForNewNodeBestLongSideFit(width, height, score2, score1); break;
		case RectBestAreaFit: newNode = FindPositionForNewNodeBestAreaFit(width, height, score1, score2); break;
	}

	if (newNode.height == 0)
		return newNode;

	newNode.cookie = cookie;
	PlaceRect(newNode);

	return newNode;
}

void MaxRectsBinPack::Insert(std::vector<RectSize> rects, std::vector<Rect> &dst, FreeRectChoiceHeuristic method)
{
	dst.clear();

	while(rects.size() > 0)
	{
		double bestScore1 = std::numeric_limits<double>::max();
		double bestScore2 = std::numeric_limits<double>::max();
		double bestRectIndex = -1;
		Rect bestNode;

		for(size_t i = 0; i < rects.size(); ++i)
		{
			double score1;
			double score2;
			Rect newNode = ScoreRect(rects[i].width, rects[i].height, method, score1, score2);

			if (score1 < bestScore1 || (score1 == bestScore1 && score2 < bestScore2))
			{
				bestScore1 = score1;
				bestScore2 = score2;
				bestNode = newNode;
				bestRectIndex = i;
			}
		}

		if (bestRectIndex == -1)
			return;

		bestNode.cookie = rects[bestRectIndex].cookie;
		PlaceRect(bestNode);
		dst.push_back(bestNode);
		rects.erase(rects.begin() + bestRectIndex);
	}
}

void MaxRectsBinPack::PlaceRect(const Rect &node)
{
	size_t numRectanglesToProcess = freeRectangles.size();
	for(size_t i = 0; i < numRectanglesToProcess; ++i)
	{
		if (SplitFreeNode(freeRectangles[i], node))
		{
			freeRectangles.erase(freeRectangles.begin() + i);
			--i;
			--numRectanglesToProcess;
		}
	}

	PruneFreeList();

	usedRectangles.push_back(node);
}

Rect MaxRectsBinPack::ScoreRect(double width, double height, FreeRectChoiceHeuristic method, double &score1, double &score2) const
{
	Rect newNode;
	score1 = std::numeric_limits<double>::max();
	score2 = std::numeric_limits<double>::max();
	switch(method)
	{
	case RectBestShortSideFit: newNode = FindPositionForNewNodeBestShortSideFit(width, height, score1, score2); break;
	case RectBottomLeftRule: newNode = FindPositionForNewNodeBottomLeft(width, height, score1, score2); break;
	case RectContactPointRule: newNode = FindPositionForNewNodeContactPoint(width, height, score1);
		score1 = -score1; // Reverse since we are minimizing, but for contact podouble score bigger is better.
		break;
	case RectBestLongSideFit: newNode = FindPositionForNewNodeBestLongSideFit(width, height, score2, score1); break;
	case RectBestAreaFit: newNode = FindPositionForNewNodeBestAreaFit(width, height, score1, score2); break;
	}

	// Cannot fit the current rectangle.
	if (newNode.height == 0)
	{
		score1 = std::numeric_limits<double>::max();
		score2 = std::numeric_limits<double>::max();
	}

	return newNode;
}

/// Computes the ratio of used surface area.
float MaxRectsBinPack::Occupancy() const
{
	double usedSurfaceArea = 0;
	for(size_t i = 0; i < usedRectangles.size(); ++i)
		usedSurfaceArea += usedRectangles[i].width * usedRectangles[i].height;

	return (float)usedSurfaceArea / (binWidth * binHeight);
}

Rect MaxRectsBinPack::FindPositionForNewNodeBottomLeft(double width, double height, double &bestY, double &bestX) const
{
	Rect bestNode;
	memset(&bestNode, 0, sizeof(Rect));

	bestY = std::numeric_limits<double>::max();
	bestX = std::numeric_limits<double>::max();

	for(size_t i = 0; i < freeRectangles.size(); ++i)
	{
		// Try to place the rectangle in upright (non-flipped) orientation.
		if (freeRectangles[i].width >= width && freeRectangles[i].height >= height)
		{
			double topSideY = freeRectangles[i].y + height;
			if (topSideY < bestY || (topSideY == bestY && freeRectangles[i].x < bestX))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = width;
				bestNode.height = height;
				bestNode.rotated = false;
				bestY = topSideY;
				bestX = freeRectangles[i].x;
			}
		}
		if (binAllowFlip && freeRectangles[i].width >= height && freeRectangles[i].height >= width)
		{
			double topSideY = freeRectangles[i].y + width;
			if (topSideY < bestY || (topSideY == bestY && freeRectangles[i].x < bestX))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = height;
				bestNode.height = width;
				bestNode.rotated = true;
				bestY = topSideY;
				bestX = freeRectangles[i].x;
			}
		}
	}
	return bestNode;
}

Rect MaxRectsBinPack::FindPositionForNewNodeBestShortSideFit(double width, double height,
	double &bestShortSideFit, double &bestLongSideFit) const
{
	Rect bestNode;
	memset(&bestNode, 0, sizeof(Rect));

	bestShortSideFit = std::numeric_limits<double>::max();
	bestLongSideFit = std::numeric_limits<double>::max();

	for(size_t i = 0; i < freeRectangles.size(); ++i)
	{
		// Try to place the rectangle in upright (non-flipped) orientation.
		if (freeRectangles[i].width >= width && freeRectangles[i].height >= height)
		{
			double leftoverHoriz = abs(freeRectangles[i].width - width);
			double leftoverVert = abs(freeRectangles[i].height - height);
			double shortSideFit = min(leftoverHoriz, leftoverVert);
			double longSideFit = max(leftoverHoriz, leftoverVert);

			if (shortSideFit < bestShortSideFit || (shortSideFit == bestShortSideFit && longSideFit < bestLongSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = width;
				bestNode.height = height;
				bestNode.rotated = false;
				bestShortSideFit = shortSideFit;
				bestLongSideFit = longSideFit;
			}
		}

		if (binAllowFlip && freeRectangles[i].width >= height && freeRectangles[i].height >= width)
		{
			double flippedLeftoverHoriz = abs(freeRectangles[i].width - height);
			double flippedLeftoverVert = abs(freeRectangles[i].height - width);
			double flippedShortSideFit = min(flippedLeftoverHoriz, flippedLeftoverVert);
			double flippedLongSideFit = max(flippedLeftoverHoriz, flippedLeftoverVert);

			if (flippedShortSideFit < bestShortSideFit || (flippedShortSideFit == bestShortSideFit && flippedLongSideFit < bestLongSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = height;
				bestNode.height = width;
				bestNode.rotated = true;
				bestShortSideFit = flippedShortSideFit;
				bestLongSideFit = flippedLongSideFit;
			}
		}
	}
	return bestNode;
}

Rect MaxRectsBinPack::FindPositionForNewNodeBestLongSideFit(double width, double height,
	double &bestShortSideFit, double &bestLongSideFit) const
{
	Rect bestNode;
	memset(&bestNode, 0, sizeof(Rect));

	bestShortSideFit = std::numeric_limits<double>::max();
	bestLongSideFit = std::numeric_limits<double>::max();

	for(size_t i = 0; i < freeRectangles.size(); ++i)
	{
		// Try to place the rectangle in upright (non-flipped) orientation.
		if (freeRectangles[i].width >= width && freeRectangles[i].height >= height)
		{
			double leftoverHoriz = abs(freeRectangles[i].width - width);
			double leftoverVert = abs(freeRectangles[i].height - height);
			double shortSideFit = min(leftoverHoriz, leftoverVert);
			double longSideFit = max(leftoverHoriz, leftoverVert);

			if (longSideFit < bestLongSideFit || (longSideFit == bestLongSideFit && shortSideFit < bestShortSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = width;
				bestNode.height = height;
				bestNode.rotated = false;
				bestShortSideFit = shortSideFit;
				bestLongSideFit = longSideFit;
			}
		}

		if (binAllowFlip && freeRectangles[i].width >= height && freeRectangles[i].height >= width)
		{
			double leftoverHoriz = abs(freeRectangles[i].width - height);
			double leftoverVert = abs(freeRectangles[i].height - width);
			double shortSideFit = min(leftoverHoriz, leftoverVert);
			double longSideFit = max(leftoverHoriz, leftoverVert);

			if (longSideFit < bestLongSideFit || (longSideFit == bestLongSideFit && shortSideFit < bestShortSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = height;
				bestNode.height = width;
				bestNode.rotated = true;
				bestShortSideFit = shortSideFit;
				bestLongSideFit = longSideFit;
			}
		}
	}
	return bestNode;
}

Rect MaxRectsBinPack::FindPositionForNewNodeBestAreaFit(double width, double height,
	double &bestAreaFit, double &bestShortSideFit) const
{
	Rect bestNode;
	memset(&bestNode, 0, sizeof(Rect));

	bestAreaFit = std::numeric_limits<double>::max();
	bestShortSideFit = std::numeric_limits<double>::max();

	for(size_t i = 0; i < freeRectangles.size(); ++i)
	{
		double areaFit = freeRectangles[i].width * freeRectangles[i].height - width * height;

		// Try to place the rectangle in upright (non-flipped) orientation.
		if (freeRectangles[i].width >= width && freeRectangles[i].height >= height)
		{
			double leftoverHoriz = abs(freeRectangles[i].width - width);
			double leftoverVert = abs(freeRectangles[i].height - height);
			double shortSideFit = min(leftoverHoriz, leftoverVert);

			if (areaFit < bestAreaFit || (areaFit == bestAreaFit && shortSideFit < bestShortSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = width;
				bestNode.height = height;
				bestNode.rotated = false;
				bestShortSideFit = shortSideFit;
				bestAreaFit = areaFit;
			}
		}

		if (binAllowFlip && freeRectangles[i].width >= height && freeRectangles[i].height >= width)
		{
			double leftoverHoriz = abs(freeRectangles[i].width - height);
			double leftoverVert = abs(freeRectangles[i].height - width);
			double shortSideFit = min(leftoverHoriz, leftoverVert);

			if (areaFit < bestAreaFit || (areaFit == bestAreaFit && shortSideFit < bestShortSideFit))
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = height;
				bestNode.height = width;
				bestNode.rotated = true;
				bestShortSideFit = shortSideFit;
				bestAreaFit = areaFit;
			}
		}
	}
	return bestNode;
}

/// Returns 0 if the two intervals i1 and i2 are disjoint, or the length of their overlap otherwise.
double CommonIntervalLength(double i1start, double i1end, double i2start, double i2end)
{
	if (i1end < i2start || i2end < i1start)
		return 0;
	return min(i1end, i2end) - max(i1start, i2start);
}

double MaxRectsBinPack::ContactPointScoreNode(double x, double y, double width, double height) const
{
	double score = 0;

	if (x == 0 || x + width == binWidth)
		score += height;
	if (y == 0 || y + height == binHeight)
		score += width;

	for(size_t i = 0; i < usedRectangles.size(); ++i)
	{
		if (usedRectangles[i].x == x + width || usedRectangles[i].x + usedRectangles[i].width == x)
			score += CommonIntervalLength(usedRectangles[i].y, usedRectangles[i].y + usedRectangles[i].height, y, y + height);
		if (usedRectangles[i].y == y + height || usedRectangles[i].y + usedRectangles[i].height == y)
			score += CommonIntervalLength(usedRectangles[i].x, usedRectangles[i].x + usedRectangles[i].width, x, x + width);
	}
	return score;
}

Rect MaxRectsBinPack::FindPositionForNewNodeContactPoint(double width, double height, double &bestContactScore) const
{
	Rect bestNode;
	memset(&bestNode, 0, sizeof(Rect));

	bestContactScore = -1;

	for(size_t i = 0; i < freeRectangles.size(); ++i)
	{
		// Try to place the rectangle in upright (non-flipped) orientation.
		if (freeRectangles[i].width >= width && freeRectangles[i].height >= height)
		{
			double score = ContactPointScoreNode(freeRectangles[i].x, freeRectangles[i].y, width, height);
			if (score > bestContactScore)
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = width;
				bestNode.height = height;
				bestNode.rotated = false;
				bestContactScore = score;
			}
		}
		if (freeRectangles[i].width >= height && freeRectangles[i].height >= width)
		{
			double score = ContactPointScoreNode(freeRectangles[i].x, freeRectangles[i].y, height, width);
			if (score > bestContactScore)
			{
				bestNode.x = freeRectangles[i].x;
				bestNode.y = freeRectangles[i].y;
				bestNode.width = height;
				bestNode.height = width;
				bestNode.rotated = true;
				bestContactScore = score;
			}
		}
	}
	return bestNode;
}

bool MaxRectsBinPack::SplitFreeNode(Rect freeNode, const Rect &usedNode)
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
			freeRectangles.push_back(newNode);
		}

		// New node at the bottom side of the used node.
		if (usedNode.y + usedNode.height < freeNode.y + freeNode.height)
		{
			Rect newNode = freeNode;
			newNode.y = usedNode.y + usedNode.height;
			newNode.height = freeNode.y + freeNode.height - (usedNode.y + usedNode.height);
			freeRectangles.push_back(newNode);
		}
	}

	if (usedNode.y < freeNode.y + freeNode.height && usedNode.y + usedNode.height > freeNode.y)
	{
		// New node at the left side of the used node.
		if (usedNode.x > freeNode.x && usedNode.x < freeNode.x + freeNode.width)
		{
			Rect newNode = freeNode;
			newNode.width = usedNode.x - newNode.x;
			freeRectangles.push_back(newNode);
		}

		// New node at the right side of the used node.
		if (usedNode.x + usedNode.width < freeNode.x + freeNode.width)
		{
			Rect newNode = freeNode;
			newNode.x = usedNode.x + usedNode.width;
			newNode.width = freeNode.x + freeNode.width - (usedNode.x + usedNode.width);
			freeRectangles.push_back(newNode);
		}
	}
	// std::cout << "\n\tSplit freeNode " << freeNode.width << " " << freeNode.height << " at " << freeNode.width << " " << freeNode.y << std::endl;

	return true;
}

void MaxRectsBinPack::PruneFreeList()
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

	/// Go through each pair and remove any rectangle that is redundant.
	for(size_t i = 0; i < freeRectangles.size(); ++i) {
		for(size_t j = i+1; j < freeRectangles.size(); ++j)
		{
			if (IsContainedIn(freeRectangles[i], freeRectangles[j]))
			{
				freeRectangles.erase(freeRectangles.begin()+i);
				--i;
				break;
			}
			if (IsContainedIn(freeRectangles[j], freeRectangles[i]))
			{
				freeRectangles.erase(freeRectangles.begin()+j);
				--j;
			}
		}
	}
}

}
