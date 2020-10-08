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
