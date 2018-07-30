#if GPU_ENABLED

#include <GL/glew.h>
#include "osr_init.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <gbm.h>
#include <vector>

namespace {

using std::string;

const char* client_exts = nullptr;
bool mesa_platform = false;
bool nvidia_platform = false;
std::vector<int> opended_drifds;

/*
 * Request OpenGL 4.3
 */
class IEglContexter {
public:
	virtual ~IEglContexter()
	{
	}

	virtual EGLDisplay createDisplay(int device_idx) const = 0;
	virtual const EGLint* getConfigAttribs() const = 0;
	virtual EGLSurface getSurfaceForCreation(EGLDisplay, EGLConfig) const = 0;
	virtual const char* name() const = 0;

	EGLContext createContext(EGLDisplay dpy, EGLContext share_context = EGL_NO_CONTEXT)
	{
	}
};

class NullContexter : public IEglContexter {
public:
	virtual ~NullContexter() {}

	virtual EGLDisplay createDisplay(int device_idx) const override
	{
		return EGL_NO_DISPLAY;
	}
	virtual const EGLint* getConfigAttribs() const override
	{
		return nullptr;
	}
	virtual EGLSurface getSurfaceForCreation(EGLDisplay, EGLConfig) const override
	{
		return EGL_NO_SURFACE;
	}
	virtual const char* name() const override
	{
		return "INVALID EGL";
	}
} null_contexter;

class NvidiaContexter : public IEglContexter {
public:
	virtual ~NvidiaContexter() {}
	virtual EGLDisplay createDisplay(int device_idx) const override
	{
		EGLDeviceEXT devices[32];
		EGLint num_devices;
		auto eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
		eglQueryDevicesEXT(32, devices, &num_devices);
		std::cerr << "NVIDIA total devices: " << num_devices << std::endl;
		auto getPlatformDisplay = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
		return getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[device_idx], NULL);
	}

	virtual const EGLint* getConfigAttribs() const override
	{
		static const EGLint nvidia_config_attribs[] = {
			EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
			EGL_BLUE_SIZE, 8,
			EGL_GREEN_SIZE, 8,
			EGL_RED_SIZE, 8,
			EGL_DEPTH_SIZE, 8,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_NONE
		};
		return nvidia_config_attribs;
	}

	virtual EGLSurface getSurfaceForCreation(EGLDisplay dpy, EGLConfig egl_cfg) const override
	{
		/*
		 * A small pbuffer as placeholder. This is apparently required
		 * by NVIDIA.
		 */
		static const EGLint pbuffer_attribs[] = {
			EGL_WIDTH, 2,
			EGL_HEIGHT, 2,
			EGL_NONE,
		};
		EGLSurface egl_surf = EGL_NO_SURFACE;
		if (egl_surf == EGL_NO_SURFACE)
			egl_surf = eglCreatePbufferSurface(dpy, egl_cfg,
							   pbuffer_attribs);
		return egl_surf;
	}

	virtual const char* name() const override
	{
		return "NVIDIA EGL";
	}
} nvidia_contexter;

class MesaContexter : public IEglContexter {
public:
	virtual ~MesaContexter() {}

	virtual EGLDisplay createDisplay(int device_idx) const override
	{
		string dripath = "/dev/dri/renderD" + std::to_string(128 + device_idx);
		int fd = open(dripath.data(), O_RDWR | O_CLOEXEC);
		std::cerr << "FD: " << fd << std::endl;
		opended_drifds.emplace_back(fd);
		gbm_device* gbm = gbm_create_device(fd);
		std::cerr << "gbm: " << gbm << std::endl;
		return eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);
	}

	virtual const EGLint* getConfigAttribs() const override
	{
		static const EGLint mesa_config_attribs[] = {
			// EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, mesa does not acccept PBuffer.
			EGL_BLUE_SIZE, 8,
			EGL_GREEN_SIZE, 8,
			EGL_RED_SIZE, 8,
			EGL_DEPTH_SIZE, 8,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_NONE
		};
		return mesa_config_attribs;
	}

	virtual EGLSurface getSurfaceForCreation(EGLDisplay, EGLConfig) const override
	{
		// Mesa does not need a surface to create EGL context.
		return EGL_NO_SURFACE;
	}

	virtual const char* name() const override
	{
		return "Mesa EGL";
	}
} mesa_contexter;

IEglContexter *contexter = &null_contexter;

}

namespace osr {

void init()
{
	client_exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
	std::cerr << "Client EXTs " << client_exts << std::endl;
	if (strstr(client_exts, "EGL_MESA_platform_gbm"))
		contexter = &mesa_contexter;
	if (strstr(client_exts, "EGL_EXT_platform_device"))
		contexter = &nvidia_contexter;
	std::cerr << "Use " << contexter->name() << " to create EGL Context\n";
}

EGLDisplay create_display(int device_idx)
{
	return contexter->createDisplay(device_idx);
}

EGLContext create_gl_context(EGLDisplay dpy, EGLContext share_context)
{
	// Requesting OpenGL 4.3
	static const EGLint ctx_attribs[] = {
		EGL_CONTEXT_MAJOR_VERSION, 4,
		EGL_CONTEXT_MINOR_VERSION, 3,
		EGL_NONE
	};

	eglInitialize(dpy, nullptr, nullptr);

	EGLint num_configs;
	EGLConfig egl_cfg;
	eglChooseConfig(dpy, contexter->getConfigAttribs(), &egl_cfg, 1, &num_configs);

	eglBindAPI(EGL_OPENGL_API);

	EGLContext core_ctx = eglCreateContext(dpy,
	                                       egl_cfg,
	                                       share_context,
	                                       ctx_attribs);
	auto egl_surf = contexter->getSurfaceForCreation(dpy, egl_cfg);
	eglMakeCurrent(dpy, egl_surf, egl_surf, core_ctx);

	glewExperimental = true;
	auto glew_err = glewInit();
	if (glew_err != GLEW_OK) {
		std::cerr << "glew init fails for dpy: " << dpy << std::endl;
		std::cerr << "Reason: " << glewGetErrorString(glew_err) << std::endl;
		// exit(EXIT_SUCCESS);
	}
	std::cerr << "glew initialized for dpy : " << dpy << std::endl;
	return core_ctx;
}

void shutdown()
{
	for (auto fd : opended_drifds) {
		close(fd);
	}
	opended_drifds.clear();
}

}

#endif // GPU_ENABLED
