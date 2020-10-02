/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef VECIN_H
#define VECIN_H

#include <string>
#include <Eigen/Core>

namespace vecio { 
	void text_read(const std::string& fn, Eigen::VectorXd& );
};

#endif
