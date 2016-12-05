#ifndef SPACE_H
#define SPACE_H

#include <cmath>

template<int ND, typename FLOAT>
class TranslationWithEulerAngleGroup {
	typedef Eigen::Matrix<FLOAT, ND, 1> Coord;
	static constexpr int TaitBryanThetaIndex = 3;
public:
	static Coord transist(const Coord& center, const Coord& delta)
	{
		// FIXME: check the correctness
		Coord ret = center;
		ret += delta;
		if (ret(TaitBryanThetaIndex) > M_PI/2.0) {
			ret(TaitBryanThetaIndex) = M_PI - ret(TaitBryanThetaIndex);
			ret(TaitBryanThetaIndex + 1) += M_PI;
			ret(TaitBryanThetaIndex + 2) += M_PI;
		} else if (ret(TaitBryanThetaIndex) < -M_PI/2.0) {
			ret(TaitBryanThetaIndex) = - ret(TaitBryanThetaIndex) - M_PI;
			ret(TaitBryanThetaIndex + 1) += M_PI;
			ret(TaitBryanThetaIndex + 2) += M_PI;
		}
		round_into_2pi(ret(TaitBryanThetaIndex + 1));
		round_into_2pi(ret(TaitBryanThetaIndex + 2));
		return ret;
	}

	static void round_into_2pi(double& val)
	{
		int r(std::floor(val / (2 * M_PI)));
		val -= r * (2 * M_PI);
	}
};

#endif
