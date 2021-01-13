/*
ReVisE: Remote visualization environment for large datasets
Copyright (C) 2021 Stepan Orlov, Alexey Kuzin, Alexey Zhuravlev, Vyacheslav Reshetnikov, Egor Usik, Vladislav Kiev, Andrey Pyatlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/agpl-3.0.en.html.

*/

#include <GL/glew.h>    // install libglew-dev
#include "OpenGlSetup.hpp"
#include "gl_check.hpp"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <map>

using namespace std;

#define EGL_ERRORTEXT_ENTRY(entry) { entry, #entry }

const char *eglErrorString(EGLenum error) {
    static map<EGLenum, const char*> errorStrings = {
        EGL_ERRORTEXT_ENTRY(EGL_BAD_ACCESS),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_ALLOC),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_ATTRIBUTE),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_CONFIG),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_CONTEXT),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_CURRENT_SURFACE),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_DISPLAY),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_MATCH),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_NATIVE_PIXMAP),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_NATIVE_WINDOW),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_PARAMETER),
        EGL_ERRORTEXT_ENTRY(EGL_BAD_SURFACE),
        EGL_ERRORTEXT_ENTRY(EGL_NON_CONFORMANT_CONFIG),
        EGL_ERRORTEXT_ENTRY(EGL_NOT_INITIALIZED),
        EGL_ERRORTEXT_ENTRY(EGL_SUCCESS)
    };
    auto it = errorStrings.find(error);
    if (it == errorStrings.end())
        return nullptr;
    else
        return it->second;
}

#undef EGL_ERRORTEXT_ENTRY

template<class ResultType>
bool checkEglResult(
        const char *file, unsigned int line,
        const char *code, ResultType result, ResultType bad_result)
{
    if (result != bad_result) {
        cout << file << ":" << line << ": " << code << " succeeded" << endl;
        return true;
    }
    else {
        auto error = eglGetError();
        auto errorString = eglErrorString(error);
        if (!errorString)
            errorString = "unknown EGL error";
        cout << file << ":" << line << ": " << code << " failed" << endl
             << "Error generated: " << error
             << " (" << errorString << ")"
             << endl;
        return false;
    }
}

#define EGL_CHECK_BOOL(egl_call) checkEglResult(__FILE__, __LINE__, #egl_call, egl_call, EGLBoolean(EGL_FALSE))
#define EGL_CHECK(egl_call, bad_result) checkEglResult(__FILE__, __LINE__, #egl_call, egl_call, bad_result)

OpenGlSetup::OpenGlSetup(int renderBufferWidth, int renderBufferHeight, bool createFbo) :
    m_renderBufferWidth(renderBufferWidth),
    m_renderBufferHeight(renderBufferHeight)
{
    /*
    Display* display = XOpenDisplay(":0");
    cout << "XOpenDisplay returned: '" << (size_t)display << "'" << endl;
    m_eglDisplay = eglGetDisplay(display);
*/
    // https://devblogs.nvidia.com/egl-eye-opengl-visualization-without-x-server/
    // https://stackoverflow.com/questions/28491665/how-to-egl-offscreen-render-to-an-image-on-linux
    //
    // 16.07.2020
    // https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/

    // Step 1 - Get the default display.
    m_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    cout << "eglGetDisplay returned "
         << (m_eglDisplay == EGL_NO_DISPLAY? "EGL_NO_DISPLAY": "value other than EGL_NO_DISPLAY")
         << endl;
    cout << "eglGetDisplay returned: '" << (size_t)m_eglDisplay << "'" << endl;

    // Step 2 - Initialize EGL.
    if (EGL_CHECK_BOOL(eglInitialize(m_eglDisplay, &m_major, &m_minor)))
        cout << "EGL implementation version is " << m_major << "." << m_minor << endl;

    static const EGLint pi32ConfigAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    EGLint pbufferAttribs[] = {
        EGL_WIDTH, renderBufferWidth,
        EGL_HEIGHT, renderBufferHeight,
        EGL_NONE
    };

    // Step 5 - Find a config that matches all requirements.
    int iConfigs = 0;
    EGL_CHECK_BOOL(eglChooseConfig(m_eglDisplay, pi32ConfigAttribs, &m_eglConfig, 1, &iConfigs));

    if (iConfigs != 1)
        throw runtime_error("Error: eglChooseConfig(): config not found.");

    // Step 6 - Create a surface to draw to.
    if (createFbo)
        m_eglSurface = EGL_NO_SURFACE;     // Doesn't work with VL
    else {
        // eglSurface = eglCreateWindowSurface(m_eglDisplay, m_eglConfig, (EGLNativeWindowType)NULL, NULL);
        EGL_CHECK(m_eglSurface = eglCreatePbufferSurface(m_eglDisplay, m_eglConfig, pbufferAttribs), EGL_NO_SURFACE);
    }

    EGL_CHECK_BOOL(eglBindAPI(EGL_OPENGL_API));

    // Step 7 - Create a context.
    EGL_CHECK(m_eglContext = eglCreateContext(m_eglDisplay, m_eglConfig, EGL_NO_CONTEXT, nullptr), EGL_NO_CONTEXT);

    // Step 8 - Bind the context to the current thread
    EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext));

    if (createFbo) {
        auto glewErr = glewInit();
        if (glewErr != GLEW_OK && glewErr != 4 /*GLEW_ERROR_NO_GLX_DISPLAY*/) {
            throw runtime_error(string("glewInit() failed") + reinterpret_cast<const char*>(glewGetErrorString(glewErr)));
        }

        // create a framebuffer object
        GL_CHECK(glGenFramebuffers(1, &m_fboId));
        GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, m_fboId));

        GL_CHECK(glGenRenderbuffers(1, &m_depthRenderbuffer));
        GL_CHECK(glGenRenderbuffers(1, &m_renderBuffer));

        doResizeRenderBuffers();
        GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depthRenderbuffer));
        GL_CHECK(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_renderBuffer));

        // check FBO status
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE) {
            ostringstream oss;
            oss << "Problem with OpenGL framebuffer after specifying color render buffer: " << status;
            throw runtime_error(oss.str());
        }
    }

    GL_CHECK(glViewport(0, 0, m_renderBufferWidth, m_renderBufferHeight));
}

void OpenGlSetup::resizeRenderBuffers(int renderBufferWidth, int renderBufferHeight)
{
    if (!(m_renderBufferWidth == renderBufferWidth && m_renderBufferHeight == renderBufferHeight)) {
        m_renderBufferWidth = renderBufferWidth;
        m_renderBufferHeight = renderBufferHeight;
        if (m_eglSurface == EGL_NO_SURFACE)
            doResizeRenderBuffers();
        else {
            GL_CHECK(eglDestroySurface(m_eglDisplay, m_eglSurface));
            EGLint pbufferAttribs[] = {
                EGL_WIDTH, renderBufferWidth,
                EGL_HEIGHT, renderBufferHeight,
                EGL_NONE
            };
            m_eglSurface = eglCreatePbufferSurface(m_eglDisplay, m_eglConfig, pbufferAttribs);
            if (m_eglSurface == EGL_NO_SURFACE)
                throw runtime_error("eglCreatePbufferSurface() failed on resize");
            eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext);
        }
        GL_CHECK(glViewport(0, 0, m_renderBufferWidth, m_renderBufferHeight));
    }
}

void OpenGlSetup::doResizeRenderBuffers()
{
    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, m_depthRenderbuffer));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, m_renderBufferWidth, m_renderBufferHeight));

    GL_CHECK(glBindRenderbuffer(GL_RENDERBUFFER, m_renderBuffer));
    GL_CHECK(glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, m_renderBufferWidth, m_renderBufferHeight));
}

OpenGlSetup::~OpenGlSetup()
{
    // Terminate EGL when finished
    eglTerminate(m_eglDisplay);
}
