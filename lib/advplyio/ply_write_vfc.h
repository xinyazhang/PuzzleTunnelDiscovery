/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef VERT_COLOR_PLY_WRITE_PLY_VC_H
#define VERT_COLOR_PLY_WRITE_PLY_VC_H

#include <Eigen/Core>
#include <string>

void ply_write_naive_header(std::ostream& fout, size_t vn, size_t fn);
void ply_write_vfc(const std::string& fn,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& C);

#endif
