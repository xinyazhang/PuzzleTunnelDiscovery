/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "convex.h"
#include <iostream>

namespace omplaux {

struct ConvexAdapter::Private {
	std::unique_ptr<fcl::Convex<double>> convex;

	std::vector<Eigen::Vector3d> V;
	std::vector<int> polygon_strip;
};

ConvexAdapter::ConvexAdapter()
{
}

ConvexAdapter::~ConvexAdapter()
{
}

ConvexAdapter::ConvexAdapter(const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F)
{
	adapt(V, F, *this);
}

void ConvexAdapter::adapt(const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		ConvexAdapter& cvx)
{
	cvx.p_.reset(new Private);
	Private& priv = *cvx.p_;
	for (int r = 0; r < F.rows(); r++) {
		priv.polygon_strip.emplace_back(F.cols());
		for (int c = 0; c < F.cols(); c++)
			priv.polygon_strip.emplace_back(F(r,c));
	}
	for (int i = 0; i < V.rows(); i++)
		priv.V.emplace_back(V.row(i));
	/*
	 * Note: no indication shows fcl is using the leading three arguments
	 */
	priv.convex.reset(new fcl::Convex<double>(nullptr, nullptr, F.rows(),
				priv.V.data(),
				priv.V.size(),
				priv.polygon_strip.data()));
}

const fcl::Convex<double>&
ConvexAdapter::getFCL() const
{
	return *p_->convex;
}

fcl::Convex<double>&
ConvexAdapter::getFCL()
{
	return *p_->convex;
}

}
