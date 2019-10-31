#include <osr/osr_render.h>
#include <osr/osr_init.h>
#include <osr/pngimage.h>
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
#if 0
	std::ofstream fout("osrsc_worker.png");
	renderer.render_depth_to(fout);
	fout.close();
#else
	renderer.views.resize(1,2);
	renderer.views << 0.0, 0.0;
	renderer.render_mvrgbd();
	osr::writePNG("osrsc_worker.png",
		      renderer.pbufferWidth, renderer.pbufferHeight,
		      renderer.mvrgb.data());
	printf("MAIN THREAD WRITTEN osrsc_worker.png\n");
#endif
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
	renderer.views.resize(1,2);
	renderer.views << 0.0, 0.0;
	// renderer.render_depth_to(fout);
	renderer.render_mvrgbd();
	printf("MAIN THREAD RENDERING %d %d\n", renderer.mvrgb.size());
	osr::writePNG("osrsc.png",
		      renderer.pbufferWidth, renderer.pbufferHeight,
		      renderer.mvrgb.data());
	renderer.render_mvrgbd(osr::Renderer::NORMAL_RENDERING);
	osr::writePNG("osrsc_1.png",
		      renderer.pbufferWidth, renderer.pbufferHeight,
		      renderer.mvrgb.data());
	printf("MAIN THREAD WRITTEN osrsc_1.png\n");
	osr::Renderer::RMMatrixXb normalmap;
	normalmap = ((renderer.mvnormal.array() + 1.0f) * 255.0f).cast<uint8_t>().matrix();
	printf("MAIN THREAD normalmap size %lu\n", normalmap.size());
	osr::writePNG("osrsc_1n.png",
		      renderer.pbufferWidth, renderer.pbufferHeight,
		      normalmap.data());
	printf("MAIN THREAD TESTING FINISHED\n");
	std::thread t1(worker, argv);

	t1.join();
	printf("WORKER THREAD TESTING FINISHED\n");
	renderer.teardown();
	osr::shutdown();
	printf("SHUTDOWN\n");
	return 0;
}
