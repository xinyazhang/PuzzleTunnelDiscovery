#include <osr/osr_render.h>
#include "quickgl.h"
#include <osr/osr_init.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <glm/ext.hpp>
#include <thread>

void worker(char* argv[])
{
	auto dpy = osr::create_display();
	if (dpy == EGL_NO_DISPLAY) {
		printf("WORKER THREAD CANNOT CREATE DISPLAY\n");
	}
	osr::create_gl_context(dpy);
	osr::Renderer renderer;
	renderer.setup();
	renderer.loadModelFromFile(argv[1]);
	renderer.angleModel(0.0f, 0.0f);
	std::ofstream fout("osrsc_worker.bin");
	renderer.render_depth_to(fout);
	fout.close();
}


int main(int argc, char *argv[])
{
	osr::init();
	auto dpy = osr::create_display();
	if (dpy == EGL_NO_DISPLAY) {
		return -1;
	}
	osr::create_gl_context(dpy);
	osr::Renderer renderer;
	renderer.setup();
	renderer.loadModelFromFile(argv[1]);
	renderer.angleModel(0.0f, 0.0f);
	std::ofstream fout("osrsc.bin");
	renderer.render_depth_to(fout);
	fout.close();
	printf("MAIN THREAD TESTING FINISHED\n");
	std::thread t1(worker, argv);

	t1.join();
	printf("WORKER THREAD TESTING FINISHED\n");
	renderer.teardown();
	osr::shutdown();
	printf("SHUTDOWN\n");
	return 0;
}
