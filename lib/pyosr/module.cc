#include <osr/osr_render.h>
#include <osr/osr_init.h>
#include <osr/gtgenerator.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <stdint.h>

namespace py = pybind11;

namespace {
uintptr_t create_display_wrapper(int device_idx = 0)
{
	EGLDisplay dpy = osr::create_display(device_idx);
	std::cerr << "dpy: " << dpy << std::endl;
	std::cerr << "sizeof dpy: " << sizeof(dpy) << std::endl;
	return (uintptr_t)dpy;
}

void create_gl_context_wrapper(uintptr_t dpy)
{
	osr::create_gl_context((EGLDisplay)dpy);
}

}

// For older pybind11
// Replace this with PYBIND11_MODULE(osr, m) after pybind11 v2.2.0
PYBIND11_PLUGIN(pyosr) {
	py::module m("pyosr", "Off-Screen Rendering");
	m.def("init", &osr::init, "Initialize OSR");
#if 1
	m.def("create_display",
	      &osr::create_display,
	      "Create EGL Display",
	      py::arg("device_idx") = 0);
	m.def("create_gl_context", &osr::create_gl_context,
	      "Create OpenGL 3.3 Core Profile Context",
	      py::arg("dpy"),
	      py::arg("share_context") = EGL_NO_CONTEXT
	      );
#else
	m.def("create_display",
	      &create_display_wrapper,
	      "Create EGL Display",
	      py::arg("device_idx") = 0);
	m.def("create_gl_context", &create_gl_context_wrapper,
	      "Create OpenGL 3.3 Core Profile Context");
#endif
	m.def("shutdown", &osr::shutdown,
	      "Close internally opened FDs. "
	      "User is responsible to free OpenGL/EGL resources");
#if 0
	m.def("transit", &osr::transitState,
	      "State Transition, given action and corresponding magnitude",
	      py::arg("state"),
	      py::arg("action"),
	      py::arg("mag"));
#endif
#if 0
	py::class_<Pet>(m, "Pet")
		.def(py::init<const std::string &>())
		.def("setName", &Pet::setName)
		.def("getName", &Pet::getName);
#endif
	using osr::Renderer;
	py::class_<Renderer>(m, "Renderer")
		.def(py::init<>())
		.def("setup", &Renderer::setup)
		.def("setupFrom", &Renderer::setupFrom)
		.def("teardown", &Renderer::teardown)
		.def("loadModelFromFile", &Renderer::loadModelFromFile)
		.def("loadRobotFromFile", &Renderer::loadRobotFromFile)
		.def("angleModel", &Renderer::angleModel)
		.def("scaleToUnit", &Renderer::scaleToUnit)
		.def_property("state", &Renderer::getRobotState, &Renderer::setRobotState)
		.def_property_readonly("scene_matrix", &Renderer::getSceneMatrix)
		.def_property_readonly("robot_matrix", &Renderer::getRobotMatrix)
		.def("is_valid_state", &Renderer::isValid)
		.def("is_disentangled", &Renderer::isDisentangled)
		.def("transit_state", &Renderer::transitState)
		.def("translate_to_unit_state", &Renderer::translateToUnitState)
		.def("render_depth_to_buffer", &Renderer::render_depth_to_buffer)
		.def("render_mvdepth_to_buffer", &Renderer::render_mvdepth_to_buffer)
		.def("render_mvrgbd", &Renderer::render_mvrgbd)
		.def_readwrite("pbufferWidth", &Renderer::pbufferWidth)
		.def_readwrite("pbufferHeight", &Renderer::pbufferHeight)
		.def_readwrite("default_depth", &Renderer::default_depth)
		.def_readwrite("mvrgb", &Renderer::mvrgb)
		.def_readwrite("mvdepth", &Renderer::mvdepth)
		.def_readwrite("views", &Renderer::views);
	using osr::GTGenerator;
	py::class_<GTGenerator>(m, "GTGenerator")
		.def(py::init<Renderer&>())
		.def("load_roadmap_file", &GTGenerator::loadRoadMapFile)
		.def("generate_gt_path", &GTGenerator::generateGTPath)
		.def_readwrite("verify_magnitude", &GTGenerator::verify_magnitude)
		.def_readwrite("gamma", &GTGenerator::gamma)
		.def_readwrite("rl_stepping_size", &GTGenerator::rl_stepping_size)
		;
	m.def("interpolate", &osr::interpolate,
	      "Interpolate between two SE3 states");
	return m.ptr();
}
