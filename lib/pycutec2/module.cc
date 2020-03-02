#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <stdint.h>

#if 0
namespace cute_c2 {
#include <cute_c2/cute_c2.h>
}
#endif
#include "pycutec2.h"

namespace py = pybind11;

PYBIND11_MODULE(pycutec2, m) {
#if 0
	using namespace cute_c2;
	m.attr("C2_MAX_POLYGON_VERTS") = py::int_(int(C2_MAX_POLYGON_VERTS));
	py::class_<c2v>(m, "c2v")
		.def(py::init<>())
		.def_readwrite("x", &c2v::x)
		.def_readwrite("y", &c2v::y)
		;
#endif
	m.def("line_segment_intersect_with_mesh",
	      &pycutec2::line_segment_intersect_with_mesh,
	      "Intersect between line segment and mesh composed by (V,E)");
	m.def("line_segments_intersect_with_mesh",
	      &pycutec2::line_segments_intersect_with_mesh,
	      "batch version of line_segment_intersect_with_mesh");
	m.def("build_mesh_2d",
	      pycutec2::build_mesh_2d);
	m.def("save_obj_1",
	      pycutec2::save_obj_1);
}
