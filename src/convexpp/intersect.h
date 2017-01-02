#ifndef INTERSECT_H
#define INTERSECT_H

#include <string>

void mesh_intersect_out(const std::string&, unsigned int p,
		double* points1, unsigned npoints1,
		int* triangles1, unsigned ntriangles1,
		double* points2, unsigned npoints2,
		int* triangles2, unsigned ntriangles2);

#endif
