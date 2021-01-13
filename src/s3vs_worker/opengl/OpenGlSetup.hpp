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

#pragma once

#include <EGL/egl.h>
#include <utility>

class OpenGlSetup
{
public:
    OpenGlSetup(int renderBufferWidth, int renderBufferHeight, bool createFbo = false);

    ~OpenGlSetup();

    EGLDisplay eglDisplay() const {
        return m_eglDisplay;
    }
    std::pair<EGLint, EGLint> eglVersion() const {
        return { m_major, m_minor };
    }
    void resizeRenderBuffers(int renderBufferWidth, int renderBufferHeight);
    int renderBufferWidth() const {
        return m_renderBufferWidth;
    }
    int renderBufferHeight() const {
        return m_renderBufferHeight;
    }
    EGLConfig eglConfig() const {
        return m_eglConfig;
    }
    EGLContext eglContext() const {
        return m_eglContext;
    }
    GLuint fboId() const {
        return m_fboId;
    }
    GLuint renderBuffer() const {
        return m_renderBuffer;
    }
    GLuint depthRenderbuffer() const {
        return m_depthRenderbuffer;
    }

private:
    int m_renderBufferWidth;
    int m_renderBufferHeight;
    EGLDisplay m_eglDisplay = nullptr;
    EGLint m_major = 0;
    EGLint m_minor = 0;
    EGLConfig m_eglConfig = nullptr;
    EGLSurface m_eglSurface = EGL_NO_SURFACE;
    EGLContext m_eglContext = nullptr;
    GLuint m_fboId = 0;
    GLuint m_renderBuffer = 0;
    GLuint m_depthRenderbuffer = 0;

    void doResizeRenderBuffers();
};
