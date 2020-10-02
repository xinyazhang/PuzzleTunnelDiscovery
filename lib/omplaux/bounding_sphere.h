/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef OMPLAUX_BOUNDING_SPHERE_H
#define OMPLAUX_BOUNDING_SPHERE_H

#include <Eigen/Core>

namespace omplaux {
void getBoundingSphere(const Eigen::MatrixXd& V,
                       Eigen::VectorXd& center,
                       double& radius);
}

#endif
