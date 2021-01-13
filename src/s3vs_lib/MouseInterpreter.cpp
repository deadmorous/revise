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

#include "MouseInterpreter.hpp"
#include "CameraController.hpp"

using namespace s3dmm;

namespace s3vs
{

MouseInterpreter::MouseInterpreter(
        const SyncVsControllerInput& input,
        CameraController &cameraController,
        const CameraTransformSetter& cameraTransformSetter) :
    m_input(input),
    m_cameraController(cameraController),
    m_cameraTransformSetter(cameraTransformSetter)
{
}

void MouseInterpreter::interpretMouse(const MouseState& mouseState)
{
    trackClickTime(mouseState);
    switch (mouseState.flags) {
    case MouseState::LeftButton:
        if (mouseMoved(mouseState) && (m_prevMouseState.flags & MouseState::LeftButton))
            rotate(mouseState);
        break;
    case MouseState::MiddleButton:
        if (mouseMoved(mouseState) && (m_prevMouseState.flags & MouseState::MiddleButton))
            pan(mouseState);
        break;
    case MouseState::RightButton:
    case MouseState::RightButton + MouseState::ShiftKey:
        if (mouseMoved(mouseState) && (m_prevMouseState.flags & MouseState::RightButton))
            zoomByMouseMove(mouseState);
        break;
    case 0:
        if (mouseDoubleClicked()) {
            resetCameraTransform();
            break;
        }
        // Intentionally falling through
    case MouseState::ShiftKey:
        if (mouseState.wheelDelta != 0)
            zoomByMouseWheel(mouseState);
        break;
    }
    m_prevMouseState = mouseState;
}

void MouseInterpreter::trackClickTime(const MouseState& mouseState)
{
    m_mouseClicked = m_mouseDoubleClicked = false;
    auto now = std::chrono::steady_clock::now();
    if (m_prevMouseState.flags == 0 && mouseState.flags == MouseState::LeftButton) {
        // Left mouse button has been pressed
        m_prevPrevLButtonDown = m_prevLButtonDown;
        m_prevLButtonDown = now;
    }
    else if (mouseState.flags != 0 && mouseState.flags != MouseState::LeftButton) {
        // Something has happened that destroys the click
        auto longAgo = now - std::chrono::seconds(10);
        m_prevLButtonDown = m_prevPrevLButtonDown = longAgo;
    }
    else if (m_prevMouseState.flags == MouseState::LeftButton && mouseState.flags == 0) {
        // Left mouse button has been released
        constexpr std::chrono::milliseconds maxClickDuration(300);
        constexpr std::chrono::milliseconds maxDoubleClickDuration(500);
        m_mouseClicked =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - m_prevLButtonDown)
            < maxClickDuration;
        m_mouseDoubleClicked =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - m_prevPrevLButtonDown)
            < maxDoubleClickDuration;
    }
}

void MouseInterpreter::rotate(const MouseState& mouseState)
{
    m_cameraController.rotate(mouseDr(mouseState));
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

void MouseInterpreter::rotateInPlane(const MouseState& mouseState)
{
    m_cameraController.rotateInPlane(mouseDr(mouseState));
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

void MouseInterpreter::pan(const MouseState& mouseState)
{
    m_cameraController.pan(mouseDr(mouseState));
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

void MouseInterpreter::zoomByMouseWheel(const MouseState &mouseState)
{
    constexpr auto ShiftZoomFactor = make_real(0.1);
    auto delta = make_real(mouseState.wheelDelta)/120;
    if (mouseState.flags & MouseState::ShiftKey)
        delta *= make_real(ShiftZoomFactor);
    m_cameraController.zoom(mousePos(mouseState), delta);
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

void MouseInterpreter::zoomByMouseMove(const MouseState& mouseState)
{
    constexpr auto ShiftZoomFactor = make_real(0.1);
    auto delta = mouseDr(mouseState)[1] / make_real(100);
    if (mouseState.flags & MouseState::ShiftKey)
        delta *= make_real(ShiftZoomFactor);
    m_cameraController.zoom(mousePos(mouseState), delta);
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

void MouseInterpreter::resetCameraTransform() {
    m_cameraController.resetCameraTransform();
    m_cameraTransformSetter(m_cameraController.cameraTransform());
}

Vec2r MouseInterpreter::mousePos(const MouseState& mouseState) const
{
    return {
        make_real(mouseState.x),
        make_real(m_input.access()->viewportSize()[1] - mouseState.y)
    };
}

Vec2r MouseInterpreter::mouseDr(const MouseState& mouseState) const
{
    return {
        make_real(mouseState.x - m_prevMouseState.x),
       -make_real(mouseState.y - m_prevMouseState.y)
    };
}

bool MouseInterpreter::mouseMoved(const MouseState& mouseState) const
{
    return !(mouseState.x == m_prevMouseState.x && mouseState.y == m_prevMouseState.y);
}

bool MouseInterpreter::mouseClicked() const {
    return m_mouseClicked;
}

bool MouseInterpreter::mouseDoubleClicked() const {
    return m_mouseDoubleClicked;
}

} // namespace s3vs
