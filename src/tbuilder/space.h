/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef SPACE_H
#define SPACE_H

#include <cmath>

template<int ND, typename FLOAT>
class TranslationWithEulerAngleGroup {
	typedef Eigen::Matrix<FLOAT, ND, 1> Coord;
	static constexpr int TaitBryanThetaIndex = 3;
	static constexpr int ThetaIndex = 3; // Roll
	static constexpr int PhiIndex = 4; // Pitch
	static constexpr int PsiIndex = 5; // Yaw
public:
	// [-Pi, Pi)
	static void round_into_2pi(double& angle)
	{
		// The least error-prone way ...
		// (Maybe also faster in our cases?)
		// FIXME: smarter way to do this.
		while (angle < -M_PI)
			angle += 2*M_PI;
		while (angle >= M_PI)
			angle -= 2*M_PI;
	}

	static Coord transist(const Coord& center, const Coord& delta)
	{
		// FIXME: check the correctness
		Coord ret = center;
		ret += delta;
		double theta = ret(ThetaIndex);
		double phi = ret(PhiIndex);
		double psi = ret(PsiIndex);
		if (phi > M_PI/2.0) {
			phi = M_PI - phi;
			ret(TaitBryanThetaIndex + 1) += M_PI;
			ret(TaitBryanThetaIndex + 2) += M_PI;
		} else if (phi < -M_PI/2.0) {
			phi = - phi - M_PI;
			ret(TaitBryanThetaIndex + 1) += M_PI;
			ret(TaitBryanThetaIndex + 2) += M_PI;
		}

		round_into_2pi(theta);
		round_into_2pi(psi);
		ret(ThetaIndex) = theta;
		ret(PhiIndex) = phi;
		ret(PsiIndex) = psi;
		return ret;
	}
};

#endif
