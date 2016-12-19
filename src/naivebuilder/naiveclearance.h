#ifndef OMPLAUX_CLEARANCE_H
#define OMPLAUX_CLEARANCE_H

#include <string>
#include <random>
#include <iostream>
#include <math.h>
#include "geo.h"

class NaiveClearance {
private:
	const Geo &env_;
	using Scalar = typename BV::S;
public:
	NaiveClearance(Geo& env)
		:env_(env)
	{
	}

	Eigen::VectorXd getCertainCube(const Eigen::Vector2d& state, bool &isfree)
	{
		// TODO
	}

	void setC(double min, double max)
	{
		min_tr_ = min;
		max_tr_ = max;
		csize_ << max_tr_ - min_tr_;
	}

	Eigen::Vector2d getCSize() const
	{
		return csize_;
	}

protected:
	double min_tr_, max_tr_;
	Eigen::Vector2d csize_; // translation size, rotation size
};
#endif
