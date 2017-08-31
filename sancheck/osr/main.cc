#include <osr/osr_init.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
	osr::init();
	auto dpy = osr::create_display();
	if (dpy == EGL_NO_DISPLAY) {
		return -1;
	}
	osr::create_gl_context(dpy);
	osr::shutdown();
	printf("PASSED\n");
	return 0;
}
