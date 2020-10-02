/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef SPACE_H
#define SPACE_H

#include <cmath>

template<int ND, typename FLOAT>
class TranslationOnlySpace {
	typedef Eigen::Matrix<FLOAT, ND, 1> Coord;
	static constexpr int TaitBryanThetaIndex = 3;
public:
	static Coord transist(const Coord& center, const Coord& delta)
	{
		return center + delta;
	}
};

#endif
