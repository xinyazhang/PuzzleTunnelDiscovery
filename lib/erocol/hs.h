/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ERODE_COLLISION_HIERARCHICAL_STRUCTURE_H
#define ERODE_COLLISION_HIERARCHICAL_STRUCTURE_H

#include <memory>
#include <vector>
#include <omplaux/geo.h>

namespace erocol {

double sqrt3 = 1.73205080756887729352744634150587;
constexpr double invsqrt3 = 0.577350269189625764509148780502;

class HModels {
	struct ColldeModel;
	struct Private;
public:
	using Transform3 = Eigen::Transform<double, 3, Eigen::AffineCompact>;

	HModels(const Geo& rob,
		const Geo& env,
		double dtr,
		double dalpha);
	~HModels();

	double getDiscretePD(const Transform3& tf);
protected:
	double getMarginForLevel(int level);
	ColldeModel& getModelAtLevel(int level);
	bool collideAtLevel(const Transform3& tf, int level);
	std::vector<ColldeModel> models_per_level_;

	const Geo &rob_;
	const Geo &env_;
	double maxr_;
	double dtr_, dalpha_;
	double dtrsqrt3_, maxrdalpha_;
	std::unique_ptr<Private> p_;

	std::string cache_dir_;
};

};

#endif
