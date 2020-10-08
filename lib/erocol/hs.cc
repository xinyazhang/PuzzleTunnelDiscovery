/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "hs.h"
#include <fat/fat.h>
#include <fcl/fcl.h>
#include <fcl/narrowphase/detail/traversal/collision_node.h>
#include <omplaux/bvh_helper.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <string>

namespace erocol {

using Scalar = double;
constexpr double kMaxLevel = 30;

struct HModels::ColldeModel {
	using BV = fcl::OBBRSS<double>;
	using BVHModel = fcl::BVHModel<BV>;

	BVHModel model;

	ColldeModel(const Geo& rob, double margin)
	{
		init(rob, margin);
	}

	ColldeModel()
	{
		inited = false;
	}

	void init(const Geo& rob, double margin, std::string fn = std::string())
	{
		Eigen::MatrixXf OV;
		Eigen::MatrixXi OF;
		fat::mkfatter(rob.V.cast<float>(), rob.F, -margin, OV, OF, true, 32.0);
		if (!fn.empty()) {
			igl::writeOBJ(fn, OV, OF);
		}
		if (OV.rows() == 0) {
			vanished = true;
		} else {
			initBVH(model,
				new fcl::detail::BVSplitter<BV>(fcl::detail::SPLIT_METHOD_MEDIAN),
				OV.cast<double>(),
				OF);
		}
		inited = true;
	}

	bool inited;
	bool vanished = false;
};

struct HModels::Private {
	HModels::ColldeModel::BVHModel env_bvh;
};

HModels::HModels(const Geo& rob, const Geo& env, double dtr, double dalpha)
	:rob_(rob),
	 env_(env),
	 dtr_(dtr),
	 dalpha_(dalpha),
	 p_(new Private)
{
	initBVH(p_->env_bvh,
		new fcl::detail::BVSplitter<ColldeModel::BV>(fcl::detail::SPLIT_METHOD_MEDIAN),
		env.V,
		env.F);

	const Eigen::MatrixXd& RV = rob_.V;
	maxr_ = 0.0;
	for (int i = 0; i < RV.rows(); i++) {
		Eigen::Vector3d v = Eigen::Vector3d(RV.row(i)) - rob_.center;
		double r = v.squaredNorm();
		maxr_ = std::max(r, maxr_);
	}
	maxr_ = std::sqrt(maxr_);
	dtrsqrt3_ = dtr_ * sqrt3;
	maxrdalpha_ = maxr_ * dalpha_;

	getModelAtLevel(0); // Init models_per_level_
}

HModels::~HModels()
{
}

double HModels::getDiscretePD(const HModels::Transform3& tf)
{
	int max_collide_level = -1;
	int level = models_per_level_.size() - 1;
	while (!collideAtLevel(tf, level) && level < kMaxLevel) {
		max_collide_level = level;
		level++;
	}
	if (level >= kMaxLevel)
		return getMarginForLevel(kMaxLevel); // Bound depth to kMaxLevel
	if (max_collide_level >= 0)
		return getMarginForLevel(max_collide_level);
	int maxlevel = models_per_level_.size() - 1;
	int minlevel = 0;
	level = (maxlevel + minlevel) / 2;
	while (maxlevel - minlevel > 2) {
		if (collideAtLevel(tf, level))
			maxlevel = level;
		else
			minlevel = level;
		level = (maxlevel + minlevel) / 2;
	}
	return getMarginForLevel(maxlevel);
}

double HModels::getMarginForLevel(int level)
{
	double dom = (1 << level);
	double factor = 1.0 / dom;
	return factor * (2.5 * maxrdalpha_ + dtrsqrt3_);
}

bool HModels::collideAtLevel(const Transform3& tf, int level)
{
	ColldeModel& cm = getModelAtLevel(level);
	if (cm.vanished)
		return false; // Success automatically.
	fcl::CollisionRequest<Scalar> request;
	fcl::CollisionResult<Scalar> result;
	fcl::collide(&cm.model, tf, &p_->env_bvh, Transform3::Identity(), request, result);
	return result.isCollision();
}

HModels::ColldeModel& HModels::getModelAtLevel(int level)
{
	if (level >= models_per_level_.size())
		models_per_level_.resize(level + 1);
#if 0
	std::cerr << "models_per_level_ size: " << models_per_level_.size() << std::endl;
	std::cerr << "models_per_level_ inited status:\n\t";
	for (const auto& m: models_per_level_)
		std::cerr << m.inited << " ";
	std::cerr << std::endl;
#endif
	if (!models_per_level_[level].inited) {
		std::cerr << "Building erode model on level " << level << " ...";
		models_per_level_[level].init(rob_, getMarginForLevel(level), "erode."+std::to_string(level)+".obj");
		std::cerr << "DONE\n\tlevel " << level << " inited: " << models_per_level_[level].inited << std::endl;
	} else {
		// std::cerr << "Level " << level << " has been inited\n";
	}
	return models_per_level_[level];
}

};
