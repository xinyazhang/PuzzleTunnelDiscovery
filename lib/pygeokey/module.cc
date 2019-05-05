#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <stdint.h>

#include <libgeokey/KeyPoint.h>
#include <libgeokey/KeySampler.h>

namespace py = pybind11;

// For older pybind11
// Replace this with PYBIND11_MODULE(osr, m) after pybind11 v2.2.0
PYBIND11_MODULE(pygeokey, m) {
	using namespace geokeyconf;
	py::class_<KeyPointProber>(m, "KeyPointProber")
		.def(py::init<const std::string&>())
		.def("probe_key_points", &KeyPointProber::probeKeyPoints)
		.def_property("alpha", &KeyPointProber::getAlpha, &KeyPointProber::setAlpha)
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
		;
}
