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

#include "s3vs/MouseState.hpp"
#include "s3vs/VsControllerInput.hpp"

namespace s3vs
{

class CameraController;

class MouseInterpreter
{
public:
    using CameraTransformSetter = std::function<void(const Matrix4r&)>;

    MouseInterpreter(
            const SyncVsControllerInput& input,
            CameraController& cameraController,
            const CameraTransformSetter& cameraTransformSetter);

    void interpretMouse(const MouseState& mouseState);

private:
    SyncVsControllerInput m_input;
    CameraController& m_cameraController;
    CameraTransformSetter m_cameraTransformSetter;

    MouseState m_prevMouseState;
    std::chrono::steady_clock::time_point m_prevLButtonDown;
    std::chrono::steady_clock::time_point m_prevPrevLButtonDown;
    bool m_mouseClicked = false;
    bool m_mouseDoubleClicked = false;
    void trackClickTime(const MouseState& mouseState);

    void rotate(const MouseState& mouseState);
    void rotateInPlane(const MouseState& mouseState);
    void pan(const MouseState& mouseState);
    void zoomByMouseWheel(const MouseState& mouseState);
    void zoomByMouseMove(const MouseState& mouseState);
    void resetCameraTransform();

    Vec2r mousePos(const MouseState& mouseState) const;
    Vec2r mouseDr(const MouseState& mouseState) const;
    bool mouseMoved(const MouseState& mouseState) const;
    bool mouseClicked() const;
    bool mouseDoubleClicked() const;
};

} // namespace s3vs
