/**
 * Copyright (C) 2020 The University of Texas at Austin
 * SPDX-License-Identifier: BSD-3-Clause or GPL-2.0-or-later
 */
#ifndef VECIO_MATIO_H
#define VECIO_MATIO_H

#include <istream>
#include <Eigen/Core>

namespace vecio {

template<typename Scalar>
std::istream& read(std::istream& fin, Eigen::MatrixBase<Scalar>& m)
{
	for(int i = 0; i < m.rows(); i++)
		for (int j = 0; j < m.cols(); j++)
			fin >> m(i,j);
	return fin;
}


}

#endif
