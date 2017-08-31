#include "osr_init.h"
#include <string.h>
#include <stdint.h>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>
#include <vector>

namespace {

using std::string;

const char* client_exts = nullptr;
bool mesa_platform = false;
bool nvidia_platform = false;
std::vector<int> opended_drifds;

EGLDisplay nvidia_create_display(int device_idx)
{
	EGLDeviceEXT devices[32];
	EGLint num_devices;
	auto eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
	eglQueryDevicesEXT(32, devices, &num_devices);
	auto getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
	return getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[device_idx], NULL);
}

EGLDisplay mesa_create_display(int device_idx)
{
	string dripath = "/dev/dri/renderD" + std::to_string(128 + device_idx);
	int fd = open(dripath.data(), O_RDWR | O_CLOEXEC);
	opended_drifds.emplace_back(fd);
	gbm_device* gbm = gbm_create_device(fd);
	return eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);
}

const EGLint config_attribs[] = {
	EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
	EGL_BLUE_SIZE, 8,
	EGL_GREEN_SIZE, 8,
	EGL_RED_SIZE, 8,
	EGL_DEPTH_SIZE, 8,
	EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
	EGL_NONE
};    

/*
 * Create a small pbuffer for placeholder.
 */
static const EGLint pbuffer_attribs[] = {
	EGL_WIDTH, 2,
	EGL_HEIGHT, 2,
	EGL_NONE,
};

/*
 * Request OpenGL 3.3
 */
static const EGLint ctx_attribs[] = {
	EGL_CONTEXT_MAJOR_VERSION, 3,
	EGL_CONTEXT_MINOR_VERSION, 3,
	EGL_NONE
};

}

namespace osr {

void init()
{
	client_exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	if (strstr(client_exts, "EGL_MESA_platform_gbm"))
		mesa_platform = true;
	if (strstr(client_exts, "EGL_EXT_platform_device"))
		nvidia_platform = true;
}

EGLDisplay create_display(int device_idx)
{
	if (mesa_platform)
		return mesa_create_display(device_idx);
	if (nvidia_platform)
		return nvidia_create_display(device_idx);
	return EGL_NO_DISPLAY;
}

void create_gl_context(EGLDisplay dpy)
{
	eglInitialize(dpy, nullptr, nullptr);

	EGLint num_configs;
	EGLConfig egl_cfg;
	eglChooseConfig(dpy, config_attribs, &egl_cfg, 1, &num_configs);

	EGLSurface egl_surf = EGL_NO_SURFACE;
	if (nvidia_platform)
		egl_surf = eglCreatePbufferSurface(dpy, egl_cfg, 
				                  pbuffer_attribs);
	eglBindAPI(EGL_OPENGL_API);

	EGLContext core_ctx = eglCreateContext(dpy,
			egl_cfg,
			EGL_NO_CONTEXT,
			ctx_attribs);
	if (nvidia_platform) {
		eglMakeCurrent(dpy, egl_surf, egl_surf, core_ctx);
	}
	if (mesa_platform) {
		eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, core_ctx);
	}
}

void shutdown()
{
	for (auto fd : opended_drifds) {
		close(fd);
	}
	opended_drifds.clear();
}

}
