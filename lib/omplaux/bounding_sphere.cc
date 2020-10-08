/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "bounding_sphere.h"

namespace omplaux {

int furthesePointIndex(const Eigen::MatrixXd& V, const Eigen::VectorXd& center)
{
	int fi = 0;
	double fdsq = (V.row(fi) - center).squaredNorm();
	for (int r = 1; r < V.rows(); r++) {
		double dsq = (V.row(r) - center).squaredNorm();
		if (dsq > fdsq) {
			fdsq = dsq;
			fi = r;
		}
	}
	return fi;
}

int getFailedVertexIndex(const Eigen::MatrixXd& V, const Eigen::VectorXd& center, double radius)
{
	double r2 = radius * radius;
	for (int r = 0; r < V.rows(); r++) {
		double dsq = (V.row(r) - center).squaredNorm();
		if (dsq > r2)
			return r;
	}
	return -1;
}

void getBoundingSphere(const Eigen::MatrixXd& V,
                       Eigen::VectorXd& center,
                       double& radius)
{
	int xi = 0;
	int yi = furthesePointIndex(V, V.row(xi));
	int zi = furthesePointIndex(V, V.row(yi));
	center = (V.row(yi) + V.row(zi)) / 2.0;
	radius = (V.row(yi) - center).norm();
	int failidx;
	while ((failidx = getFailedVertexIndex(V, center, radius)) >= 0) {
		Eigen::VectorXd failvec = V.row(failidx) - center;
		failvec *= 1.0 + radius / failvec.norm();
		center = (V.row(failidx) + V.row(failidx) + failvec) / 2.0;
		radius = failvec.norm();
	}
}

}
