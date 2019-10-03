#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <stdint.h>

#include <libgeokey/KeyPoint.h>
#include <libgeokey/KeySampler.h>

namespace py = pybind11;

PYBIND11_MODULE(pygeokey, m) {
	using namespace geokeyconf;
	py::class_<KeyPointProber>(m, "KeyPointProber")
		.def(py::init<const std::string&>())
		.def("probe_key_points", &KeyPointProber::probeKeyPoints,
		     py::arg("attempts"),
		     py::arg("seed") = 0
		    )
		.def("probe_notch_points", &KeyPointProber::probeNotchPoints,
		     py::arg("seed") = 0,
		     py::arg("keep_intermediate_data") = false
		    )
		.def("get_intermediate_data", &KeyPointProber::getIntermediateData)
		.def("get_all_skeleton_points", &KeyPointProber::getAllSkeletonPoints)
		.def("get_skeleton_edges", &KeyPointProber::getSkeletonEdges)
		.def_property("alpha", &KeyPointProber::getAlpha, &KeyPointProber::setAlpha)
		.def_property("local_min_thresh", &KeyPointProber::getLocalMinThresh, &KeyPointProber::setLocalMinThresh)
		.def_property("group_tolerance_epsilon", &KeyPointProber::getToleranceEpsilon, &KeyPointProber::setToleranceEpsilon)
		;
	py::class_<KeySampler>(m, "KeySampler")
		.def(py::init<const std::string&, const std::string&>())
		.def("get_all_key_configs", &KeySampler::getAllKeyConfigs)
		.def("get_key_configs", &KeySampler::getKeyConfigs,
		     py::arg("env_key_point"),
		     py::arg("rob_key_point"),
		     py::arg("number_of_rotations_env"),
		     py::arg("number_of_rotations_rob") = 1
		    )
		.def_property("thresh_narrow_tunnel_ratio", &KeySampler::getThreshNarrowTunnelRatio, &KeySampler::setThreshNarrowTunnelRatio)
		;
}
