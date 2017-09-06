#include <osr/osr_init.h>
#include <osr/osr_render.h>
#include <pybind11/pybind11.h>
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
	      "Create OpenGL 3.3 Core Profile Context");
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
	return m.ptr();
#if 0
	py::class_<Pet>(m, "Pet")
		.def(py::init<const std::string &>())
		.def("setName", &Pet::setName)
		.def("getName", &Pet::getName);
#endif
}
