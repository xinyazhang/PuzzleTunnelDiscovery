/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef VECIO_ARRAY_VEC_IO_H
#define VECIO_ARRAY_VEC_IO_H

#include <ostream>
#include <vector>
#include <Eigen/Core>

std::ostream& operator<<(std::ostream& fout, const std::vector<Eigen::VectorXd>& milestones);

#endif
