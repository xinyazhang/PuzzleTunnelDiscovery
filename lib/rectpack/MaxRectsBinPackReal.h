/**
 * SPDX-FileCopyrightText: Copyright Â© 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
/** @file MaxRectsBinPack.h
	@author Jukka Jylänki

	@brief Implements different bin packer algorithms that use the MAXRECTS data structure.

	This work is released to Public Domain, do whatever you want with it.
*/
#pragma once

#include <vector>
#include <memory>

#include "RectReal.h"

namespace rbp {

class FreeRectangleManager;

/** MaxRectsBinPack implements the MAXRECTS data structure and different bin packing algorithms that 
	use this structure. */
class MaxRectsBinPack
{
public:
	/// Instantiates a bin of size (0,0). Call Init to create a new bin.
	MaxRectsBinPack();

	/// Instantiates a bin of the given size.
	/// @param allowFlip Specifies whether the packing algorithm is allowed to rotate the input rectangles by 90 degrees to consider a better placement.
	MaxRectsBinPack(double width, double height, bool allowFlip = true);

	/// (Re)initializes the packer to an empty bin of width x height units. Call whenever
	/// you need to restart with a new bin.
	void Init(double width, double height, bool allowFlip = true);

	/// Specifies the different heuristic rules that can be used when deciding where to place a new rectangle.
	enum FreeRectChoiceHeuristic
	{
		RectBestShortSideFit, ///< -BSSF: Positions the rectangle against the short side of a free rectangle into which it fits the best.
		RectBestLongSideFit, ///< -BLSF: Positions the rectangle against the long side of a free rectangle into which it fits the best.
		RectBestAreaFit, ///< -BAF: Positions the rectangle into the smallest free rect into which it fits.
		RectBottomLeftRule, ///< -BL: Does the Tetris placement.
		RectContactPointRule ///< -CP: Choosest the placement where the rectangle touches other rects as much as possible.
	};

	/// Inserts the given list of rectangles in an offline/batch mode, possibly rotated.
	/// @param rects The list of rectangles to insert. This vector is passed by value because internally it will be destroyed in the process.
	/// @param dst [out] This list will contain the packed rectangles. The indices will not correspond to that of rects.
	/// @param method The rectangle placement rule to use when packing.
	void Insert(std::vector<RectSize> rects, std::vector<Rect> &dst, FreeRectChoiceHeuristic method);

	/// Inserts a single rectangle into the bin, possibly rotated.
	Rect Insert(double width, double height, FreeRectChoiceHeuristic method, void *cookie = nullptr);

	/// Computes the ratio of used surface area to the total bin area.
	float Occupancy() const;

private:
	double binWidth;
	double binHeight;

	bool binAllowFlip;

	std::vector<Rect> usedRectangles;

	std::shared_ptr<FreeRectangleManager> frm_;

	/// Computes the placement score for placing the given rectangle with the given method.
	/// @param score1 [out] The primary placement score will be outputted here.
	/// @param score2 [out] The secondary placement score will be outputted here. This isu sed to break ties.
	/// @return This struct identifies where the rectangle would be placed if it were placed.
	Rect ScoreRect(double width, double height, FreeRectChoiceHeuristic method, double &score1, double &score2) const;

	/// Places the given rectangle into the bin.
	void PlaceRect(const Rect &node);

	/// Computes the placement score for the -CP variant.
	double ContactPointScoreNode(double x, double y, double width, double height) const;

	Rect FindPositionForNewNodeBottomLeft(double width, double height, double &bestY, double &bestX) const;
	Rect FindPositionForNewNodeBestShortSideFit(double width, double height, double &bestShortSideFit, double &bestLongSideFit) const;
	Rect FindPositionForNewNodeBestLongSideFit(double width, double height, double &bestShortSideFit, double &bestLongSideFit) const;
	Rect FindPositionForNewNodeBestAreaFit(double width, double height, double &bestAreaFit, double &bestShortSideFit) const;
	Rect FindPositionForNewNodeContactPoint(double width, double height, double &contactScore) const;

};

}
