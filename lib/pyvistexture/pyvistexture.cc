#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "texture_viewer.h"

namespace py = pybind11;

class PyTextureViewer : public TextureViewer {
public:
	virtual bool key_up(unsigned int key, int modifier) override
	{
		PYBIND11_OVERLOAD(bool,               /* Return type */
		                  TextureViewer,      /* Parent class */
		                  key_up,             /* Name of function in C++ (must match Python name) */
		                  key, modifier       /* Argument(s) */
		                  );
	}
};

PYBIND11_MODULE(pyvistexture, m) {
	py::class_<TextureViewer, PyTextureViewer>(m, "TextureViewer")
		.def(py::init<>())
		.def("load_geometry", &TextureViewer::loadGeometry)
		.def("init_viewer", &TextureViewer::initViewer)
		.def("update_geometry", &TextureViewer::updateGeometry)
		.def("update_texture", &TextureViewer::updateTexture)
		.def("update_point_cloud", &TextureViewer::updatePointCloud)
		.def("run", &TextureViewer::run)
		.def("key_up", &TextureViewer::key_up)
		;
}
