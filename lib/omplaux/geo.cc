#include "geo.h"

#include <igl/readOBJ.h>
#include <igl/per_vertex_normals.h>

void Geo::read(const std::string& fn)
{
	igl::readOBJ(fn, V, F);
	//center << 0.0, 0.0, 0.0; // Origin
	center = V.colwise().mean().cast<double>();
	//center << 16.973146438598633, 1.2278236150741577, 10.204807281494141;
	// From OMPL.app, no idea how they get this.
	GPUV = V.cast<float>();
	igl::per_vertex_normals(GPUV, F, N);
	std::cerr << "center: " << center << std::endl;
#if 0
	std::cerr << N << std::endl;;
#endif
}
