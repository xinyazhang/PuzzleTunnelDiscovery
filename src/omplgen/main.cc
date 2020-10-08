/**
 * SPDX-FileCopyrightText: Copyright © 2020 The University of Texas at Austin
 * SPDX-FileContributor: Xinya Zhang <xinyazhang@utexas.edu>
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <mazeinfo/2d.h>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
	if (argc < 4) {
		std::cerr << "Need more arguments" << std::endl;
		std::cerr << "Arguments: <input maze file> <output wallspace ply file> <output robot ply file>" << std::endl;
		return -1;
	}
	// Use opt.get_input_stream() instead of Options object for
	// orthogonality
	std::ifstream imaze(argv[1]);
	MazeBoundary wall(imaze);
	MazeBoundary stick(imaze);

	std::ofstream ofwall(argv[2]);
	std::ofstream ofstick(argv[3]);
	wall.writePLY(ofwall, Eigen::Vector3d(1.0f, 1.0f, 0.0f));
	stick.writePLY(ofstick, Eigen::Vector3d(0.0f, 0.0f, 1.0f));
	return 0;
}
