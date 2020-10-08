/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "vecout.h"
#include <stdexcept>
#include <iostream>
#include <fstream>

using std::endl;

void vecio::text_write(const std::string& ofn, const Eigen::VectorXd& IV)
{
	std::ostream* pfout;
	std::ofstream fout;
	if (ofn.empty()) {
		pfout = &std::cout;
	} else {
		fout.exceptions(std::ofstream::failbit | std::ofstream::badbit);
		fout.open(ofn);
		pfout = &fout;
	}
	(*pfout) << IV.rows() << endl << IV << endl;
}
