/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef OMPLAUX_CLEARANCE_H
#define OMPLAUX_CLEARANCE_H

#include <Eigen/Core>
#include <iostream>
#include <memory>
#include <omplaux/geo.h>

class NaiveClearance {
public:
	NaiveClearance(Geo& env);
	~NaiveClearance();

	Eigen::VectorXd getCertainCube(const Eigen::Vector2d& state, bool &isfree);

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

	struct NaiveClearancePrivate;
	std::unique_ptr<NaiveClearancePrivate> p_;
};
#endif
