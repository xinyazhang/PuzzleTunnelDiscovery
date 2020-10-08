/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef VECOUT_H
#define VECOUT_H

#include <string>
#include <Eigen/Core>

namespace vecio { 
	void text_write(const std::string& fn, const Eigen::VectorXd& );
};

#endif

