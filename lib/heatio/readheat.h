/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef READ_HEAT_H
#define READ_HEAT_H

#include <fstream>
#include <Eigen/Core>
#include <string>

struct HeatFrame {
	Eigen::VectorXd hvec;
	double t;
	size_t nvert;
	double sum;
};

class HeatReader {
public:
	HeatReader(std::ifstream& _fin);
	bool read_frame(HeatFrame&);
private:
	bool binary_ = false;
	std::ifstream& fin;

	void common_init();
	bool ascii_read(HeatFrame& frame);
	bool bin_read(HeatFrame& frame);
};

#endif
