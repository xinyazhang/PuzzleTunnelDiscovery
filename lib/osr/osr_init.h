#ifndef OFF_SCREEN_RENDERING_INIT_H
#define OFF_SCREEN_RENDERING_INIT_H

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
void create_gl_context(EGLDisplay dpy);
void shutdown();
}

#endif
