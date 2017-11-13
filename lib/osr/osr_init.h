#ifndef OFF_SCREEN_RENDERING_INIT_H
#define OFF_SCREEN_RENDERING_INIT_H

#if GPU_ENABLED

#include <EGL/egl.h>
#include <EGL/eglext.h>

/*
 * OSR: Off-Screen Rendering support functions.
 *      This set of functions does the necessary initialization for OSR.
 * Post-condition: a valid OpenGL context is created, but no guarntee on the
 *      framebuffer
 */
namespace osr {
void init();
EGLDisplay create_display(int device_idx = 0);
EGLContext create_gl_context(EGLDisplay dpy, EGLContext share_context = EGL_NO_CONTEXT);
void shutdown();
}

#endif // GPU_ENABLED

#endif // OFF_SCREEN_RENDERING_INIT_H
