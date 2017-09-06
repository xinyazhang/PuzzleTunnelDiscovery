#include <osr/osr_render.h>
#include "quickgl.h"
#include <osr/osr_init.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <glm/ext.hpp>


int main(int argc, char *argv[])
{
	osr::init();
	auto dpy = osr::create_display();
	if (dpy == EGL_NO_DISPLAY) {
		return -1;
	}
	osr::create_gl_context(dpy);
	osr::shutdown();
	osr::Renderer renderer;
	renderer.setup();
	renderer.loadModelFromFile(argv[1]);
	renderer.angleModel(30.0f, 30.0f);
	std::ofstream fout("osrsc.bin");
	renderer.render_depth_to(fout);
	fout.close();
	renderer.teardown();
	printf("PASSED\n");
	return 0;
}
